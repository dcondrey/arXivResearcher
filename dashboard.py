"""
arXiv Research Intelligence Dashboard
=====================================
Interactive visualization dashboard for exploring research data.

Usage:
    uv run streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import math

# Page configuration
st.set_page_config(
    page_title="arXiv Research Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_papers_data():
    """Load the main papers dataset."""
    data_path = Path("arxiv_dataset/arxiv_papers_all.csv")
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        return None

    df = pd.read_csv(data_path)

    # Parse dates
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df['year'] = df['published_date'].dt.year
    df['month'] = df['published_date'].dt.to_period('M').astype(str)
    df['year_month'] = df['published_date'].dt.strftime('%Y-%m')

    # Fill missing values
    df['citations'] = df['citations'].fillna(0).astype(int)
    df['abstract'] = df['abstract'].fillna('')
    df['title'] = df['title'].fillna('')

    # Parse author list
    df['author_list'] = df['authors'].fillna('').str.split('; ')
    df['author_count'] = df['author_list'].apply(len)

    # Parse category list
    df['category_list'] = df['all_categories'].fillna('').str.split(', ')
    df['category_count'] = df['category_list'].apply(len)

    # Calculate paper age and citation velocity
    today = datetime.now()
    df['paper_age_days'] = (today - df['published_date']).dt.days
    df['citations_per_month'] = df.apply(
        lambda x: x['citations'] / max(x['paper_age_days'] / 30, 1) if x['paper_age_days'] > 0 else 0,
        axis=1
    )

    return df


@st.cache_data
def load_statistics():
    """Load pre-computed statistics."""
    stats_path = Path("arxiv_dataset/dataset_statistics.json")
    if not stats_path.exists():
        return None

    with open(stats_path) as f:
        return json.load(f)


def get_category_description(cat):
    """Get human-readable category descriptions."""
    descriptions = {
        "cs.AI": "Artificial Intelligence",
        "cs.AR": "Hardware Architecture",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering",
        "cs.CG": "Computational Geometry",
        "cs.CL": "Computation and Language",
        "cs.CR": "Cryptography and Security",
        "cs.CV": "Computer Vision",
        "cs.CY": "Computers and Society",
        "cs.DB": "Databases",
        "cs.DC": "Distributed Computing",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DS": "Data Structures and Algorithms",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.GT": "Game Theory",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.MS": "Mathematical Software",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking",
        "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SC": "Symbolic Computation",
        "cs.SD": "Sound",
        "cs.SE": "Software Engineering",
        "cs.SI": "Social and Information Networks",
    }
    return descriptions.get(cat, cat)


# ============================================================================
# Overview Page
# ============================================================================

def render_overview_page(df, stats):
    """Render the Overview page."""
    st.markdown('<p class="main-header">Dataset Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore the arXiv research dataset at a glance</p>',
                unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Papers",
            value=f"{len(df):,}",
            delta=None
        )

    with col2:
        st.metric(
            label="Total Citations",
            value=f"{df['citations'].sum():,}",
            delta=None
        )

    with col3:
        st.metric(
            label="Categories",
            value=f"{df['primary_category'].nunique()}",
            delta=None
        )

    with col4:
        date_range = f"{df['published_date'].min().strftime('%Y-%m')} to {df['published_date'].max().strftime('%Y-%m')}"
        st.metric(
            label="Date Range",
            value=date_range,
            delta=None
        )

    st.markdown("---")

    # Charts section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Papers Over Time")
        papers_by_month = df.groupby('year_month').size().reset_index(name='count')
        papers_by_month = papers_by_month.sort_values('year_month')

        fig = px.line(
            papers_by_month,
            x='year_month',
            y='count',
            markers=True,
            labels={'year_month': 'Month', 'count': 'Number of Papers'}
        )
        fig.update_layout(
            hovermode='x unified',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Categories by Paper Count")
        category_counts = df['primary_category'].value_counts().head(15).reset_index()
        category_counts.columns = ['category', 'count']
        category_counts['description'] = category_counts['category'].apply(get_category_description)

        fig = px.bar(
            category_counts,
            y='category',
            x='count',
            orientation='h',
            labels={'category': 'Category', 'count': 'Papers'},
            hover_data=['description']
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Citation distribution and category comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Citation Distribution")
        # Use log scale for better visualization
        df_cited = df[df['citations'] > 0].copy()

        fig = px.histogram(
            df_cited,
            x='citations',
            nbins=50,
            labels={'citations': 'Citations', 'count': 'Number of Papers'},
            log_y=True
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Citations",
            yaxis_title="Number of Papers (log scale)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Citation stats
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Median Citations", f"{df['citations'].median():.0f}")
        with col_b:
            st.metric("Mean Citations", f"{df['citations'].mean():.2f}")
        with col_c:
            pct_cited = (df['citations'] > 0).mean() * 100
            st.metric("% Papers with Citations", f"{pct_cited:.1f}%")

    with col2:
        st.subheader("Average Citations by Category")
        cat_citations = df.groupby('primary_category').agg({
            'citations': 'mean',
            'arxiv_id': 'count'
        }).reset_index()
        cat_citations.columns = ['category', 'avg_citations', 'paper_count']
        cat_citations = cat_citations[cat_citations['paper_count'] >= 20]
        cat_citations = cat_citations.sort_values('avg_citations', ascending=False).head(15)

        fig = px.bar(
            cat_citations,
            y='category',
            x='avg_citations',
            orientation='h',
            labels={'category': 'Category', 'avg_citations': 'Avg Citations'},
            color='avg_citations',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top cited papers table
    st.subheader("Top Cited Papers")
    top_papers = df.nlargest(10, 'citations')[
        ['title', 'authors', 'primary_category', 'published_date', 'citations', 'arxiv_url']
    ].copy()
    top_papers['published_date'] = top_papers['published_date'].dt.strftime('%Y-%m-%d')
    top_papers.columns = ['Title', 'Authors', 'Category', 'Published', 'Citations', 'URL']

    st.dataframe(
        top_papers,
        column_config={
            "URL": st.column_config.LinkColumn("Link")
        },
        hide_index=True,
        use_container_width=True
    )


# ============================================================================
# Trends Page
# ============================================================================

def render_trends_page(df, stats):
    """Render the Trends page."""
    st.markdown('<p class="main-header">Research Trends</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze emerging and declining research areas</p>',
                unsafe_allow_html=True)

    # Category trend analysis
    st.subheader("Category Trends Over Time")

    # Calculate trends for each category
    category_monthly = df.groupby(['year_month', 'primary_category']).size().unstack(fill_value=0)

    # Allow category selection
    selected_categories = st.multiselect(
        "Select categories to compare",
        options=df['primary_category'].unique().tolist(),
        default=df['primary_category'].value_counts().head(5).index.tolist(),
        max_selections=10
    )

    if selected_categories:
        fig = go.Figure()
        for cat in selected_categories:
            if cat in category_monthly.columns:
                fig.add_trace(go.Scatter(
                    x=category_monthly.index,
                    y=category_monthly[cat],
                    mode='lines+markers',
                    name=cat,
                    hovertemplate=f'{cat}<br>%{{x}}<br>Papers: %{{y}}<extra></extra>'
                ))

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Papers",
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Emerging vs Declining categories
    st.markdown("---")
    col1, col2 = st.columns(2)

    # Calculate momentum for each category
    sorted_months = sorted(df['year_month'].unique())
    if len(sorted_months) >= 4:
        recent_months = sorted_months[-len(sorted_months)//4:]
        early_months = sorted_months[:len(sorted_months)//4]

        momentum_data = []
        for cat in df['primary_category'].unique():
            cat_df = df[df['primary_category'] == cat]
            recent_count = len(cat_df[cat_df['year_month'].isin(recent_months)])
            early_count = len(cat_df[cat_df['year_month'].isin(early_months)])

            if early_count > 0:
                momentum = (recent_count - early_count) / early_count
            else:
                momentum = 1.0 if recent_count > 0 else 0

            momentum_data.append({
                'category': cat,
                'momentum': momentum,
                'recent_papers': recent_count,
                'early_papers': early_count,
                'total_papers': len(cat_df)
            })

        momentum_df = pd.DataFrame(momentum_data)
        momentum_df = momentum_df[momentum_df['total_papers'] >= 20]  # Filter small categories

        with col1:
            st.subheader("Emerging Categories")
            emerging = momentum_df.nlargest(10, 'momentum')

            fig = px.bar(
                emerging,
                y='category',
                x='momentum',
                orientation='h',
                color='momentum',
                color_continuous_scale='Greens',
                labels={'momentum': 'Growth Rate', 'category': 'Category'}
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Declining Categories")
            declining = momentum_df.nsmallest(10, 'momentum')

            fig = px.bar(
                declining,
                y='category',
                x='momentum',
                orientation='h',
                color='momentum',
                color_continuous_scale='Reds_r',
                labels={'momentum': 'Growth Rate', 'category': 'Category'}
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total descending'},
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # Hot papers section
    st.markdown("---")
    st.subheader("Hot Papers (High Citation Velocity)")

    # Papers with high citation velocity
    hot_papers = df[df['citations_per_month'] > 0].nlargest(15, 'citations_per_month')[
        ['title', 'primary_category', 'citations', 'citations_per_month', 'paper_age_days', 'arxiv_url']
    ].copy()
    hot_papers['citations_per_month'] = hot_papers['citations_per_month'].round(2)
    hot_papers.columns = ['Title', 'Category', 'Citations', 'Citations/Month', 'Age (days)', 'URL']

    st.dataframe(
        hot_papers,
        column_config={
            "URL": st.column_config.LinkColumn("Link"),
            "Citations/Month": st.column_config.NumberColumn(format="%.2f")
        },
        hide_index=True,
        use_container_width=True
    )

    # Citation velocity by category
    st.subheader("Average Citation Velocity by Category")
    velocity_by_cat = df[df['citations'] > 0].groupby('primary_category').agg({
        'citations_per_month': 'mean',
        'arxiv_id': 'count'
    }).reset_index()
    velocity_by_cat.columns = ['category', 'avg_velocity', 'papers_with_citations']
    velocity_by_cat = velocity_by_cat[velocity_by_cat['papers_with_citations'] >= 5]
    velocity_by_cat = velocity_by_cat.sort_values('avg_velocity', ascending=False).head(15)

    fig = px.bar(
        velocity_by_cat,
        y='category',
        x='avg_velocity',
        orientation='h',
        labels={'category': 'Category', 'avg_velocity': 'Avg Citations/Month'},
        color='avg_velocity',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Opportunities Page
# ============================================================================

def render_opportunities_page(df, stats):
    """Render the Research Opportunities page."""
    st.markdown('<p class="main-header">Research Opportunities</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Identify research gaps and underexplored areas</p>',
                unsafe_allow_html=True)

    # Category intersection analysis
    st.subheader("Underexplored Category Intersections")
    st.markdown("""
    These category pairs appear together less frequently than expected,
    suggesting potential research opportunities at their intersection.
    """)

    # Count category co-occurrences
    category_counts = Counter()
    pair_counts = Counter()

    for categories in df['category_list']:
        for cat in categories:
            category_counts[cat] += 1
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                pair = tuple(sorted([cat1, cat2]))
                pair_counts[pair] += 1

    total_papers = len(df)
    underexplored = []

    for (cat1, cat2), actual in pair_counts.items():
        expected = (category_counts[cat1] * category_counts[cat2]) / total_papers
        if expected > 5:
            ratio = actual / expected
            if ratio < 0.5 and category_counts[cat1] > 20 and category_counts[cat2] > 20:
                underexplored.append({
                    'Category 1': cat1,
                    'Category 2': cat2,
                    'Actual Papers': actual,
                    'Expected Papers': round(expected, 1),
                    'Ratio': round(ratio, 3),
                    'Opportunity Score': round((1 - ratio) * min(expected, 50), 1)
                })

    underexplored_df = pd.DataFrame(underexplored)
    if not underexplored_df.empty:
        underexplored_df = underexplored_df.sort_values('Opportunity Score', ascending=False).head(20)
        st.dataframe(underexplored_df, hide_index=True, use_container_width=True)
    else:
        st.info("No significantly underexplored intersections found.")

    # Category opportunity scores
    st.markdown("---")
    st.subheader("Category Opportunity Scores")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Calculate opportunity metrics for each category
        opportunity_data = []

        sorted_months = sorted(df['year_month'].unique())
        recent_months = sorted_months[-len(sorted_months)//4:] if len(sorted_months) >= 4 else sorted_months

        for cat in df['primary_category'].unique():
            cat_df = df[df['primary_category'] == cat]
            if len(cat_df) < 20:
                continue

            avg_citations = cat_df['citations'].mean()
            avg_velocity = cat_df[cat_df['citations'] > 0]['citations_per_month'].mean() if len(cat_df[cat_df['citations'] > 0]) > 0 else 0
            recent_count = len(cat_df[cat_df['year_month'].isin(recent_months)])
            recency_rate = recent_count / len(cat_df)

            # Opportunity score components
            score = (
                min(avg_velocity * 2, 3) +
                recency_rate * 2 +
                (1 - min(len(cat_df) / 500, 1)) +
                min(avg_citations / 20, 2)
            )

            opportunity_data.append({
                'Category': cat,
                'Description': get_category_description(cat),
                'Papers': len(cat_df),
                'Avg Citations': round(avg_citations, 1),
                'Citation Velocity': round(avg_velocity, 3),
                'Growth Rate': round(recency_rate * 100, 1),
                'Opportunity Score': round(score, 2)
            })

        opp_df = pd.DataFrame(opportunity_data)
        opp_df = opp_df.sort_values('Opportunity Score', ascending=False)

        fig = px.bar(
            opp_df.head(15),
            y='Category',
            x='Opportunity Score',
            orientation='h',
            color='Opportunity Score',
            color_continuous_scale='RdYlGn',
            hover_data=['Description', 'Papers', 'Avg Citations']
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**How scores are calculated:**")
        st.markdown("""
        - **Citation Velocity**: Higher velocity = more interest
        - **Growth Rate**: Recent activity indicates emerging area
        - **Competition**: Less crowded fields score higher
        - **Impact Potential**: Based on avg citations
        """)

        st.markdown("---")
        st.markdown("**Top Recommendations:**")
        for _, row in opp_df.head(5).iterrows():
            st.markdown(f"- **{row['Category']}**: {row['Description']}")

    # Full table
    st.markdown("---")
    st.subheader("Full Category Opportunity Table")
    st.dataframe(
        opp_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Opportunity Score": st.column_config.ProgressColumn(
                min_value=0,
                max_value=opp_df['Opportunity Score'].max()
            )
        }
    )


# ============================================================================
# Paper Explorer Page
# ============================================================================

def render_paper_explorer_page(df, stats):
    """Render the Paper Explorer page."""
    st.markdown('<p class="main-header">Paper Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Search and filter the research database</p>',
                unsafe_allow_html=True)

    # Filters in sidebar
    st.sidebar.markdown("### Filters")

    # Search
    search_query = st.sidebar.text_input("Search in title/abstract", "")

    # Category filter
    categories = ['All'] + sorted(df['primary_category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Category", categories)

    # Date range
    min_date = df['published_date'].min().date()
    max_date = df['published_date'].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Citation range
    min_cites, max_cites = st.sidebar.slider(
        "Citation range",
        min_value=0,
        max_value=int(df['citations'].max()),
        value=(0, int(df['citations'].max()))
    )

    # Sort options
    sort_by = st.sidebar.selectbox(
        "Sort by",
        ["Published Date (newest)", "Published Date (oldest)",
         "Citations (highest)", "Citations (lowest)",
         "Citation Velocity (highest)"]
    )

    # Apply filters
    filtered_df = df.copy()

    if search_query:
        mask = (
            filtered_df['title'].str.lower().str.contains(search_query.lower(), na=False) |
            filtered_df['abstract'].str.lower().str.contains(search_query.lower(), na=False)
        )
        filtered_df = filtered_df[mask]

    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['primary_category'] == selected_category]

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['published_date'].dt.date >= date_range[0]) &
            (filtered_df['published_date'].dt.date <= date_range[1])
        ]

    filtered_df = filtered_df[
        (filtered_df['citations'] >= min_cites) &
        (filtered_df['citations'] <= max_cites)
    ]

    # Sort
    if sort_by == "Published Date (newest)":
        filtered_df = filtered_df.sort_values('published_date', ascending=False)
    elif sort_by == "Published Date (oldest)":
        filtered_df = filtered_df.sort_values('published_date', ascending=True)
    elif sort_by == "Citations (highest)":
        filtered_df = filtered_df.sort_values('citations', ascending=False)
    elif sort_by == "Citations (lowest)":
        filtered_df = filtered_df.sort_values('citations', ascending=True)
    elif sort_by == "Citation Velocity (highest)":
        filtered_df = filtered_df.sort_values('citations_per_month', ascending=False)

    # Display results
    st.markdown(f"**Found {len(filtered_df):,} papers**")

    # Pagination
    papers_per_page = 20
    total_pages = max(1, (len(filtered_df) + papers_per_page - 1) // papers_per_page)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

    start_idx = (page - 1) * papers_per_page
    end_idx = start_idx + papers_per_page
    page_df = filtered_df.iloc[start_idx:end_idx]

    # Display papers
    for _, paper in page_df.iterrows():
        with st.expander(f"**{paper['title']}** - {paper['citations']} citations"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Authors:** {paper['authors']}")
                st.markdown(f"**Abstract:** {paper['abstract'][:500]}..." if len(paper['abstract']) > 500 else f"**Abstract:** {paper['abstract']}")

            with col2:
                st.markdown(f"**Category:** {paper['primary_category']}")
                st.markdown(f"**Published:** {paper['published_date'].strftime('%Y-%m-%d')}")
                st.markdown(f"**Citations:** {paper['citations']}")
                st.markdown(f"**Velocity:** {paper['citations_per_month']:.2f}/mo")
                st.markdown(f"[View on arXiv]({paper['arxiv_url']})")

    st.markdown(f"Page {page} of {total_pages}")


# ============================================================================
# Author Network Page
# ============================================================================

def render_author_network_page(df, stats):
    """Render the Author Network page."""
    st.markdown('<p class="main-header">Author Network</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore author collaborations and metrics</p>',
                unsafe_allow_html=True)

    # Build author metrics
    author_metrics = defaultdict(lambda: {
        'papers': 0,
        'citations': 0,
        'coauthors': set(),
        'categories': Counter()
    })

    for _, paper in df.iterrows():
        authors = paper['author_list']
        citations = paper['citations']
        category = paper['primary_category']

        for author in authors:
            author = author.strip()
            if author:
                author_metrics[author]['papers'] += 1
                author_metrics[author]['citations'] += citations
                author_metrics[author]['categories'][category] += 1

                for coauthor in authors:
                    coauthor = coauthor.strip()
                    if coauthor and coauthor != author:
                        author_metrics[author]['coauthors'].add(coauthor)

    # Convert to DataFrame
    author_data = []
    for author, metrics in author_metrics.items():
        if metrics['papers'] >= 2:  # Filter out single-paper authors
            author_data.append({
                'Author': author,
                'Papers': metrics['papers'],
                'Citations': metrics['citations'],
                'Avg Citations': round(metrics['citations'] / metrics['papers'], 1),
                'Coauthors': len(metrics['coauthors']),
                'Primary Category': metrics['categories'].most_common(1)[0][0] if metrics['categories'] else ''
            })

    author_df = pd.DataFrame(author_data)

    # Top authors table
    st.subheader("Top Authors by Paper Count")

    col1, col2 = st.columns(2)

    with col1:
        top_by_papers = author_df.nlargest(20, 'Papers')
        fig = px.bar(
            top_by_papers,
            y='Author',
            x='Papers',
            orientation='h',
            color='Citations',
            color_continuous_scale='Blues',
            hover_data=['Avg Citations', 'Coauthors']
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=20, r=20, t=20, b=20),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Authors by Citations")
        top_by_cites = author_df.nlargest(20, 'Citations')
        fig = px.bar(
            top_by_cites,
            y='Author',
            x='Citations',
            orientation='h',
            color='Papers',
            color_continuous_scale='Greens',
            hover_data=['Avg Citations', 'Coauthors']
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=20, r=20, t=20, b=20),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    # Author search
    st.markdown("---")
    st.subheader("Author Search")

    search_author = st.text_input("Search for an author")

    if search_author:
        matching = author_df[author_df['Author'].str.lower().str.contains(search_author.lower())]
        if not matching.empty:
            st.dataframe(matching.sort_values('Citations', ascending=False), hide_index=True)
        else:
            st.info("No authors found matching your search.")

    # Collaboration network visualization (if pyvis available)
    st.markdown("---")
    st.subheader("Most Connected Authors")

    top_connected = author_df.nlargest(15, 'Coauthors')

    fig = px.scatter(
        top_connected,
        x='Papers',
        y='Coauthors',
        size='Citations',
        color='Avg Citations',
        hover_name='Author',
        labels={'Coauthors': 'Number of Coauthors', 'Papers': 'Number of Papers'},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Full author table
    st.markdown("---")
    st.subheader("Full Author Table")

    sort_col = st.selectbox(
        "Sort by",
        ['Papers', 'Citations', 'Avg Citations', 'Coauthors']
    )

    st.dataframe(
        author_df.sort_values(sort_col, ascending=False).head(100),
        hide_index=True,
        use_container_width=True
    )


# ============================================================================
# Success Factors Page
# ============================================================================

def render_success_factors_page(df, stats):
    """Render the Success Factors page."""
    st.markdown('<p class="main-header">Success Factors</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze what drives paper impact</p>',
                unsafe_allow_html=True)

    # Team size analysis
    st.subheader("Impact of Team Size")

    df['team_size_bucket'] = df['author_count'].apply(
        lambda x: 'Solo' if x == 1
        else 'Small (2-3)' if x <= 3
        else 'Medium (4-6)' if x <= 6
        else 'Large (7+)'
    )

    team_stats = df.groupby('team_size_bucket').agg({
        'citations': ['mean', 'median', 'count'],
        'citations_per_month': 'mean'
    }).round(2)
    team_stats.columns = ['Avg Citations', 'Median Citations', 'Paper Count', 'Avg Velocity']
    team_stats = team_stats.reset_index()
    team_stats.columns = ['Team Size', 'Avg Citations', 'Median Citations', 'Paper Count', 'Avg Velocity']

    # Order the categories
    team_order = ['Solo', 'Small (2-3)', 'Medium (4-6)', 'Large (7+)']
    team_stats['Team Size'] = pd.Categorical(team_stats['Team Size'], categories=team_order, ordered=True)
    team_stats = team_stats.sort_values('Team Size')

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            team_stats,
            x='Team Size',
            y='Avg Citations',
            color='Avg Citations',
            color_continuous_scale='Blues',
            text='Avg Citations'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(team_stats, hide_index=True, use_container_width=True)

    # Category cross-listing analysis
    st.markdown("---")
    st.subheader("Impact of Category Cross-listing")

    df['is_cross_listed'] = df['category_count'] > 1

    cross_list_stats = df.groupby('is_cross_listed').agg({
        'citations': ['mean', 'median', 'count']
    }).round(2)
    cross_list_stats.columns = ['Avg Citations', 'Median Citations', 'Paper Count']
    cross_list_stats = cross_list_stats.reset_index()
    cross_list_stats['is_cross_listed'] = cross_list_stats['is_cross_listed'].map({True: 'Cross-listed', False: 'Single Category'})
    cross_list_stats.columns = ['Type', 'Avg Citations', 'Median Citations', 'Paper Count']

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            cross_list_stats,
            values='Paper Count',
            names='Type',
            title='Distribution of Cross-listing'
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            cross_list_stats,
            x='Type',
            y='Avg Citations',
            color='Type',
            text='Avg Citations'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Abstract length analysis
    st.markdown("---")
    st.subheader("Impact of Abstract Length")

    df['abstract_length'] = df['abstract'].str.split().str.len().fillna(0)
    df['abstract_bucket'] = df['abstract_length'].apply(
        lambda x: 'Short (<100 words)' if x < 100
        else 'Medium (100-200)' if x < 200
        else 'Long (200-300)' if x < 300
        else 'Very Long (300+)'
    )

    abstract_stats = df.groupby('abstract_bucket').agg({
        'citations': ['mean', 'count']
    }).round(2)
    abstract_stats.columns = ['Avg Citations', 'Paper Count']
    abstract_stats = abstract_stats.reset_index()
    abstract_stats.columns = ['Abstract Length', 'Avg Citations', 'Paper Count']

    # Order categories
    length_order = ['Short (<100 words)', 'Medium (100-200)', 'Long (200-300)', 'Very Long (300+)']
    abstract_stats['Abstract Length'] = pd.Categorical(abstract_stats['Abstract Length'], categories=length_order, ordered=True)
    abstract_stats = abstract_stats.sort_values('Abstract Length')

    fig = px.bar(
        abstract_stats,
        x='Abstract Length',
        y='Avg Citations',
        color='Avg Citations',
        color_continuous_scale='Viridis',
        text='Avg Citations'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Title characteristics
    st.markdown("---")
    st.subheader("Impact of Title Characteristics")

    df['has_colon'] = df['title'].str.contains(':')
    df['has_question'] = df['title'].str.contains(r'\?')
    df['short_title'] = df['title'].str.split().str.len() <= 8

    title_analysis = []

    for feature, label in [('has_colon', 'Has Colon (:)'),
                           ('has_question', 'Has Question (?)'),
                           ('short_title', 'Short Title (<=8 words)')]:
        with_feat = df[df[feature]]['citations'].mean()
        without_feat = df[~df[feature]]['citations'].mean()
        lift = with_feat / max(without_feat, 0.1)

        title_analysis.append({
            'Feature': label,
            'Avg Citations (With)': round(with_feat, 2),
            'Avg Citations (Without)': round(without_feat, 2),
            'Lift': round(lift, 2)
        })

    title_df = pd.DataFrame(title_analysis)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(title_df, hide_index=True, use_container_width=True)

    with col2:
        fig = px.bar(
            title_df,
            x='Feature',
            y='Lift',
            color='Lift',
            color_continuous_scale='RdYlGn',
            text='Lift'
        )
        fig.add_hline(y=1, line_dash="dash", line_color="gray")
        fig.update_traces(textposition='outside')
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Key insights
    st.markdown("---")
    st.subheader("Key Insights")

    insights = []

    # Team size insight
    if not team_stats.empty:
        best_team = team_stats.loc[team_stats['Avg Citations'].idxmax()]
        insights.append(f"**Team Size**: {best_team['Team Size']} teams have the highest average citations ({best_team['Avg Citations']:.1f})")

    # Cross-listing insight
    if not cross_list_stats.empty:
        cross_avg = cross_list_stats[cross_list_stats['Type'] == 'Cross-listed']['Avg Citations'].values
        single_avg = cross_list_stats[cross_list_stats['Type'] == 'Single Category']['Avg Citations'].values
        if len(cross_avg) > 0 and len(single_avg) > 0:
            if cross_avg[0] > single_avg[0]:
                insights.append(f"**Cross-listing**: Cross-listed papers have {cross_avg[0]/max(single_avg[0], 0.1):.1f}x more citations on average")

    # Title insights
    if not title_df.empty:
        best_title = title_df.loc[title_df['Lift'].idxmax()]
        if best_title['Lift'] > 1.1:
            insights.append(f"**Title**: Papers with '{best_title['Feature']}' tend to get {best_title['Lift']:.2f}x more citations")

    for insight in insights:
        st.markdown(f"- {insight}")


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application entry point."""
    # Load data
    df = load_papers_data()
    stats = load_statistics()

    if df is None:
        st.error("Failed to load data. Please ensure the data files exist in the arxiv_dataset directory.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")

    pages = {
        "Overview": render_overview_page,
        "Trends": render_trends_page,
        "Opportunities": render_opportunities_page,
        "Paper Explorer": render_paper_explorer_page,
        "Author Network": render_author_network_page,
        "Success Factors": render_success_factors_page
    }

    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    st.sidebar.markdown(f"**Papers:** {len(df):,}")
    st.sidebar.markdown(f"**Categories:** {df['primary_category'].nunique()}")
    st.sidebar.markdown(f"**Total Citations:** {df['citations'].sum():,}")

    # Render selected page
    pages[selected_page](df, stats)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    arXiv Research Intelligence Dashboard

    Built with Streamlit and Plotly
    """)


if __name__ == "__main__":
    main()

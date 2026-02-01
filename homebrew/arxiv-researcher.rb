class ArxivResearcher < Formula
  include Language::Python::Virtualenv

  desc "Research intelligence and visualization dashboard for arXiv"
  homepage "https://github.com/dcondrey/arXivResearcher"
  url "https://github.com/dcondrey/arXivResearcher/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "Apache-2.0"
  head "https://github.com/dcondrey/arXivResearcher.git", branch: "main"

  depends_on "python@3.11"

  resource "requests" do
    url "https://files.pythonhosted.org/packages/source/r/requests/requests-2.31.0.tar.gz"
    sha256 "942c5a758f98d790eaed1a29cb6eefc7ffb0d1cf7af05c3d2791656dbd6ad1e1"
  end

  resource "pandas" do
    url "https://files.pythonhosted.org/packages/source/p/pandas/pandas-2.1.4.tar.gz"
    sha256 "fcb68203c833cc735321512e13861358079a96c174a61f5116a1de89c58c0ef7"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.3.tar.gz"
    sha256 "697df43e2b6310ecc9d95f05d5ef20eacc09c7c4ecc9da3f235d39e71b7da1e4"
  end

  resource "networkx" do
    url "https://files.pythonhosted.org/packages/source/n/networkx/networkx-3.2.1.tar.gz"
    sha256 "9f1bb5cf3409bf324e0a722c20bdb4c20ee39bf1c30ce8ae499c8502b0b5e0c6"
  end

  resource "matplotlib" do
    url "https://files.pythonhosted.org/packages/source/m/matplotlib/matplotlib-3.8.2.tar.gz"
    sha256 "01a978b871b881ee76017152f1f1a0cbf6bd5f7b8ff8c96df0df1bd57d8755a1"
  end

  resource "seaborn" do
    url "https://files.pythonhosted.org/packages/source/s/seaborn/seaborn-0.13.1.tar.gz"
    sha256 "b5e54fd4fa5f0c1e867c87a2d61f6a45b70f85c92b4b0cf1d3f1bb25bf5c2f17"
  end

  resource "openpyxl" do
    url "https://files.pythonhosted.org/packages/source/o/openpyxl/openpyxl-3.1.2.tar.gz"
    sha256 "a6f5977418ebb3f8b8ca6f5e8a1e5e5a5e5d5e5d5e5a5e5d5e5a5e5a5e5a5e5a"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    system bin/"arxiv-researcher", "--version"
  end
end

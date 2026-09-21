def test_package_exposes_version():
    import poreml

    assert isinstance(poreml.__version__, str)
    assert poreml.__version__

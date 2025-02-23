import pytest

import pandas as pd

EXPECTED_TARGET_CATEGORIES = {'setosa', 'versicolor', 'virginica'}

TARGET_COL = 'species'

def test_no_missing_values(df):
    """Ensure there is no missing values in the dataset before training."""
    assert not df.isnull().values.any(), 'data contains missing values'

def test_numerical_ranges(df):
    """Check if numerical features fall within expected ranges."""
    #@TODO repeat this for other features
    assert df['sepal_length'].between(4, 8).all(), 'sepal length out of range'

def test_label_categories(df):
    """Ensure target column only contains expected values."""
    assert set(df[TARGET_COL].unique()).issubset(EXPECTED_TARGET_CATEGORIES), 'found unexpected target value'

def run_tests(df):
    """Executes all pytest-based data checks."""
    res = pytest.main(['-q'], plugins=[DfPlugin(df)])
    return res


class DfPlugin:
    """
    A pytest plugin that injects a Pandas DataFrame into test functions that require it.

    This class allows pytest to automatically provide a DataFrame (`df`) to test functions
    that have `df` as a parameter, enabling seamless data-driven testing.
    """
    def __init__(self, df):
        """
        Initializes the DfPlugin with a given Pandas DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be passed to test functions.
        """
        self.df = df

    def pytest_generate_tests(self, metafunc):
        """
        Hooks into pytest's test generation process to provide the DataFrame as a parameter.

        If a test function has 'df' as a parameter, this method injects the stored DataFrame
        into the test execution.

        Args:
            metafunc (pytest.Metafunc): A pytest object that provides test function metadata.
        """
        if 'df' in metafunc.fixturenames:
            metafunc.parametrize('df', [self.df])

from typing import List, Dict, Any
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataTransformer:
    """
    A class to handle various data transformations including encoding and scaling.

    Attributes:
    ----------
    df : pd.DataFrame
        The input dataframe.
    features_to_remove : List[str]
        List of feature names to be removed.
    age_order : List[str]
        List defining the order of age categories for ordinal encoding.
    education_order : List[str]
        List defining the order of education categories for ordinal encoding.
    month_order : List[str]
        List defining the order of month categories for ordinal encoding.
    poutcome_order : List[str]
        List defining the order of poutcome categories for ordinal encoding.
    binary_mapping : Dict[str, int]
        Dictionary to map binary values to integers.
    columns_to_map : List[str]
        List of columns to be mapped using binary mapping.
    transformer : ColumnTransformer
        The column transformer pipeline.

    Methods:
    -------
    _create_transformer() -> ColumnTransformer:
        Creates the column transformer for encoding and scaling.
    _apply_binary_mapping() -> None:
        Applies binary mapping to specified columns.
    _get_feature_names_out() -> List[str]:
        Retrieves the output feature names after transformation.
    _remove_duplicates() -> None:
        Removes duplicate rows from the dataframe.
    _remove_unknown() -> None:
        Removes rows with unknown values in specific columns.
    _combine_basic_education() -> None:
        Combines different basic education levels into a single category.
    _label_encode() -> None:
        Applies label encoding to the contact column.
    transform() -> pd.DataFrame:
        Applies the transformations to the dataframe.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataTransformer with the dataframe.

        Parameters:
        ----------
        df : pd.DataFrame
            The input dataframe.
        """
        self.df = df

        self.features_to_remove = ["duration", "day_of_week"]
        self.age_order = ["young", "young_adult", "middle_aged", "late_middle_aged", "old_age"]
        self.education_order = ["illiterate", "education.basic", "high.school", "professional.course",
                                "university.degree"]
        self.month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        self.poutcome_order = ["nonexistent", "failure", "success"]

        self.binary_mapping = {"yes": 1, "no": 0}
        self.columns_to_map = ["default", "loan", "housing", "y"] if "y" in self.df.columns else ["default", "loan",
                                                                                                  "housing"]
        self.transformer = self._create_transformer()

    def _create_transformer(self) -> ColumnTransformer:
        """
        Creates the column transformer for encoding and scaling.

        Returns:
        -------
        ColumnTransformer
            The column transformer pipeline.
        """
        return ColumnTransformer(
            transformers=[
                # ordinal encoding
                ('bins_age', Pipeline(steps=[
                    ('ordinal', OrdinalEncoder(categories=[self.age_order])),
                    ('scaler', StandardScaler())
                ]), ['bins_age']),

                ('education', Pipeline(steps=[
                    ('ordinal', OrdinalEncoder(categories=[self.education_order])),
                    ('scaler', StandardScaler())
                ]), ['education']),

                ('month', Pipeline(steps=[
                    ('ordinal', OrdinalEncoder(categories=[self.month_order])),
                    ('scaler', StandardScaler())
                ]), ['month']),

                ('poutcome', Pipeline(steps=[
                    ('ordinal', OrdinalEncoder(categories=[self.poutcome_order])),
                    ('scaler', StandardScaler())
                ]), ['poutcome']),

                # One-Hot encoding for job and marital
                ('job_marital', OneHotEncoder(), ['job', 'marital']),
                # Standard scaling of the rest numeric features
                ('scaling', StandardScaler(), ['previous', 'campaign'])
            ],
            # leave the other columns unchanged
            remainder='passthrough',
            force_int_remainder_cols=False
        )

    def _clean_data(self):
        self._remove_feature_columns()
        self._remove_duplicates()
        self._remove_unknown()
        self._combine_basic_education()
        self._bin_age()

    def _apply_binary_mapping(self) -> None:
        """
        Applies binary mapping to specified columns.
        """
        for column in self.columns_to_map:
            self.df[column] = self.df[column].map(self.binary_mapping)

    def _remove_feature_columns(self):
        self.df.drop(self.features_to_remove, axis=1, inplace=True)

    def _remove_duplicates(self) -> None:
        """
        Removes duplicate rows from the dataframe.
        """
        self.df.drop_duplicates(inplace=True)

    def _remove_unknown(self) -> None:
        """
        Removes rows with unknown values in specific columns.
        """
        unknown_columns = ["job", "education", "default", "housing"]
        for column in unknown_columns:
            self.df = self.df[self.df[column] != "unknown"]

    def _combine_basic_education(self) -> None:
        """
        Combines different basic education levels into a single category.
        """
        self.df.loc[self.df['education'].isin(['basic.4y', 'basic.6y', 'basic.9y']), 'education'] = 'education.basic'

    def _bin_age(self):
        bins_age = pd.cut(self.df['age'], bins=len(self.age_order),
                          labels=self.age_order)  # [(16.919, 33.2] < (33.2, 49.4] < (49.4, 65.6] < (65.6, 81.8] < (81.8, 98.0]
        self.df.insert(1, 'bins_age', bins_age)
        self.df = self.df.drop('age', axis=1)

    def _label_encode(self) -> None:
        """
        Applies label encoding to the contact column.
        """
        label_encoder = LabelEncoder()
        self.df.loc[:, 'contact'] = label_encoder.fit_transform(self.df['contact'])

    def clean_and_transform(self) -> pd.DataFrame:
        """
        Applies the transformations to the dataframe.

        Returns:
        -------
        pd.DataFrame
            The transformed dataframe.
        """
        self._clean_data()
        self._label_encode()
        self._apply_binary_mapping()
        return self.df

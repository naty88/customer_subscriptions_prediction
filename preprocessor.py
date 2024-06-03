from typing import List, Dict

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataTransformer:
    def __init__(self, df: pd.DataFrame,
                 features_to_remove: List[str],
                 age_order: List[str],
                 education_order: List[str],
                 month_order: List[str],
                 poutcome_order: List[str],
                 binary_mapping: Dict[str, int],
                 columns_to_map: List[str]
                 ):
        self.df = df
        self.features_to_remove = features_to_remove
        self.age_order = age_order
        self.education_order = education_order
        self.month_order = month_order
        self.poutcome_order = poutcome_order
        self.binary_mapping = binary_mapping
        self.columns_to_map = columns_to_map
        self.transformer = self._create_transformer()

    def _create_transformer(self):
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
            remainder='passthrough'
        )

    def make_preprocess(self):
        self._remove_feature_columns()
        self._remove_duplicates()
        self._remove_unknown()
        self._combine_basic_education()
        self._label_encode()
        self._bin_age()
        self._binary_map()
        return self.df

    def _remove_feature_columns(self):
        self.df.drop(self.features_to_remove, axis=1, inplace=True)

    def _remove_duplicates(self):
        self.df.drop_duplicates(keep="last", inplace=True)

    def _remove_unknown(self):
        self.df = self.df.query(
            'job != "unknown" & education != "unknown" & default != "unknown" & housing != "unknown"')
        self.df.reset_index(inplace=True)

    def _combine_basic_education(self):
        self.df.loc[self.df['education'].isin(['basic.4y', 'basic.6y', 'basic.9y']), 'education'] = 'education.basic'

    def _label_encode(self):
        label_encoder = LabelEncoder()
        self.df.loc[:, 'contact'] = label_encoder.fit_transform(self.df['contact'])

    def _bin_age(self):
        bins_age = pd.cut(self.df['age'], bins=len(self.age_order), labels=self.age_order)
        self.df.insert(1, 'bins_age', bins_age)
        self.df = self.df.drop('age', axis=1)

    def _binary_map(self):
        for column in self.columns_to_map:
            self.df[column] = self.df[column].map(self.binary_mapping)

from typing import List, Dict

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        self.features_to_remove = ["duration", "day_of_week"]
        # hierarchical order for some ordinal features
        self.age_order = ["young", "young_adult", "middle_aged", "late_middle_aged", "old_age"]
        self.education_order = ["illiterate", "education.basic", "high.school", "professional.course",
                                "university.degree"]
        self.month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        self.poutcome_order = ["nonexistent", "failure", "success"]

        # binary mapping definitions
        self.binary_mapping = {"yes": 1, "no": 0}
        self.columns_to_map = ["default", "loan", "housing", "y"] if "y" in self.df.columns else ["default", "loan", "housing"]
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
            remainder='passthrough',
            force_int_remainder_cols=False
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
        bins_age = pd.cut(self.df['age'], bins=len(self.age_order),
                          labels=self.age_order)  # [(16.919, 33.2] < (33.2, 49.4] < (49.4, 65.6] < (65.6, 81.8] < (81.8, 98.0]
        self.df.insert(1, 'bins_age', bins_age)
        self.df = self.df.drop('age', axis=1)

    def _binary_map(self):
        for column in self.columns_to_map:
            self.df[column] = self.df[column].map(self.binary_mapping)

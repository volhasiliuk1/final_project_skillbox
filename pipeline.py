import pandas as pd
import numpy as np
import dill

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder



def data_cleaning(df):
    df = df.copy()
    m = df['device_browser'] == 'Safari'
    df.loc[m, 'device_brand'] = df.loc[m, 'device_brand'].replace(np.nan, 'Apple')
    n = df['device_browser'] == 'Samsung Internet'
    df.loc[n, 'device_brand'] = df.loc[n, 'device_brand'].replace(np.nan, 'Samsung')

    df.loc[df['utm_source'] == '(none)', 'utm_source'] = '(not set)'
    df.loc[df['device_brand'] == '(not set)', 'device_brand'] = 'other'
    df.loc[df['geo_country'] != 'Russia', 'geo_country'] = 'other'
    df.loc[df['utm_campaign'].map(df['utm_campaign'].value_counts(normalize=True).lt(0.01)), 'utm_campaign'] = 'other'
    df.loc[df['utm_source'].map(df['utm_source'].value_counts(normalize=True).lt(0.01)), 'utm_source'] = 'other'
    df.loc[df['device_browser'].str.contains('Instagram'), 'device_browser'] = 'Instagram'
    return df


def feature_engineering(df):
    df = df.copy()
    df[['screen_width', 'screen_height']] = df.device_screen_resolution.str.split('x', expand=True).astype(int)
    df['diagonal'] = np.sqrt(df.screen_width ** 2 + df.screen_height ** 2)
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df['day_of_year'] = df['visit_date'].dt.dayofyear
    df[['hour', 'minute', 'second']] = df.visit_time.str.split(':', expand=True).astype(int)
    df['total_seconds'] = df.second + df.minute * 60 + df.hour * 60 * 60

    def get_coord(city):
        geolocator = Nominatim(user_agent="my_request")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        try:
            location = geolocator.geocode(city)
            if location:
                lat = geolocator.geocode(city).latitude
                lon = geolocator.geocode(city).longitude
                return lat, lon
            else:
                return 0, 0
        except:
            return 0, 0

    latitude_array = []
    longitude_array = []
    for i in range(len(df.geo_city.unique())):
        print(f'Объект {i+1} из {len(df.geo_city.unique())}')
        city = df.geo_city.unique()[i]
        lat, lon = get_coord(city)
        latitude_array.append(lat)
        longitude_array.append(lon)
    latitude_dict = {df.geo_city.unique()[i]: latitude_array[i] for i in range(len(latitude_array))}
    longitude_dict = {df.geo_city.unique()[i]: longitude_array[i] for i in range(len(longitude_array))}

    df['latitude'] = df['geo_city'].map(latitude_dict)
    df['longitude'] = df['geo_city'].map(longitude_dict)
    return df


def smoothing_data(df):
    df = df.copy()

    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    boundaries = calculate_outliers(df['diagonal'])

    df.loc[df['diagonal'] < boundaries[0], 'diagonal'] = round(boundaries[0])
    df.loc[df['diagonal'] > boundaries[1], 'diagonal'] = round(boundaries[1])
    return df


def filter_data(df):
    df = df.copy()
    columns_to_drop = [
       'screen_width',
       'screen_height',
       'device_screen_resolution',
       'visit_date',
       'visit_time',
       'geo_city',
       'hour',
       'minute',
       'second'
    ]
    df = df.drop(columns_to_drop, axis=1)
    return df


def main():

    print('Sber Auto Prediction Pipeline')

    session = pd.read_csv('data/ga_sessions.csv').drop(columns=['device_model','utm_keyword','device_os', 'utm_adcontent'], axis=1)
    hits = pd.read_csv('data/ga_hits.csv').drop(columns=['hit_date', 'hit_time', 'hit_number', 'hit_type', 'hit_referer', 'hit_page_path', 'event_category', 'event_label', 'event_value'])
    target_event = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
              'sub_custom_question_submit_click',
              'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
              'sub_car_request_submit_click']
    hits['target'] = np.where(hits['event_action'].isin(target_event), 1, 0)
    hits = hits.drop('event_action', axis=1)
    df = session.merge(hits)
    df = df.sort_values('target', ascending=False).drop_duplicates('session_id').sort_index()
    df = df.drop(columns=['session_id', 'client_id'], axis=1)
    df = df.dropna()
    X_input = df.drop(['target'], axis=1)
    y_input = df['target']

    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X_input, y_input)

    data_preparation = Pipeline(steps=[
        ('data_cleaning', FunctionTransformer(data_cleaning)),
        ('feature_engineering', FunctionTransformer(feature_engineering)),
        ('smoothing', FunctionTransformer(smoothing_data)),
        ('filter', FunctionTransformer(filter_data))
    ])

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=['object'])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    model = GradientBoostingClassifier()


    pipe = Pipeline(steps=[
        ('preparation', data_preparation),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')

    print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
    pipe.fit(X, y)

    with open("sber_auto_pipe.pkl", "wb") as f:
        dill.dump({
            'model': pipe,
            'metadata': {

                'name': 'Sber auto prediction model',
                'author': 'Volha Krasouskaya',
                'version': 1,
                'type': type(pipe.named_steps["classifier"]).__name__,
                'roc_auc': score

            }
        }, f, recurse=True )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()


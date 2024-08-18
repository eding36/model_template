"""Python Model Example"""
import pandas as pd
import os

parent_dir = os.path.dirname(os.getcwd())
print(parent_dir)

df_health_exposure_1 = pd.read_csv(parent_dir + '/input/epr_health_exposure.csv')

df_health_exposure_2 = pd.read_csv(parent_dir + '/input/epr_health_exposure_validation.csv')

import numpy as np


#
def filter_data(df, columns_of_interest):
    important_data = df[columns_of_interest]

    important_data['he_r166_diabetes_par'] = important_data['he_r166a_diabetes_mom'] + important_data[
        'he_r166b_diabetes_dad']
    important_data['he_r167_hbp_par'] = important_data['he_r167a_hbp_mom'] + important_data['he_r167b_hbp_dad']

    important_data = important_data.drop(
        columns=['he_r166a_diabetes_mom', 'he_r166b_diabetes_dad', 'he_r167a_hbp_mom', 'he_r167b_hbp_dad'])

    important_data.replace('.M', np.nan, inplace=True)

    return important_data


columns_of_interest_training = ['epr_number', 'he_b007_hypertension_PARQ', 'he_bmi_derived', 'he_a005_physical_health',
                                'he_b009_atherosclerosis', 'he_b011_angina', 'he_b010_cardiac_arrhythmia',
                                'he_b012_heart_attack', 'he_b013_coronary_artery', 'he_b015_poor_blood_flow',
                                'he_b019_stroke_mini', 'he_b020_stroke_PARQ', 'he_c022_diabetes_PARQ',
                                'he_f045_fatty_liver', 'he_j062_bone_loss', 'he_j063_osteoporosis',
                                'he_r166a_diabetes_mom', 'he_r166b_diabetes_dad', 'he_r167a_hbp_mom',
                                'he_r167b_hbp_dad', 'he_b008_high_cholesterol']

columns_of_interest_val = ['epr_number', 'he_b007_hypertension_PARQ', 'he_bmi_derived', 'he_a005_physical_health',
                           'he_b009_atherosclerosis', 'he_b011_angina', 'he_b010_cardiac_arrhythmia',
                           'he_b012_heart_attack', 'he_b013_coronary_artery', 'he_b015_poor_blood_flow',
                           'he_b019_stroke_mini', 'he_b020_stroke_PARQ', 'he_c022_diabetes_PARQ', 'he_f045_fatty_liver',
                           'he_j062_bone_loss', 'he_j063_osteoporosis', 'he_r166a_diabetes_mom',
                           'he_r166b_diabetes_dad', 'he_r167a_hbp_mom', 'he_r167b_hbp_dad']

df_health_exposure_filtered_1 = filter_data(df_health_exposure_1, columns_of_interest_training)
df_health_exposure_filtered_2 = filter_data(df_health_exposure_2, columns_of_interest_val)

from sklearn.metrics import auc, roc_curve
import tensorflow as tf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer()

training_array_pre_impute = np.array(df_health_exposure_filtered_1)
transformed_array = imputer.fit_transform(training_array_pre_impute)
#np.random.seed(42)
#np.random.shuffle(transformed_array)
X_train = transformed_array[0:2450,list(range(1, 16)) + [17, 18]]
X_test = transformed_array[2451:,list(range(1, 16)) + [17, 18]]
y_train = np.round(transformed_array[0:2450,16])
y_test = np.round(transformed_array[2451:,16])
print(X_train.shape)

print(X_test.shape)

model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(17,)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
       # tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
      #  tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    #    tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Training Accuracy: {accuracy:.2f}")

predictions = model.predict(X_test)
fpr,tpr, thresholds = roc_curve(y_test,predictions)
roc_auc = auc(fpr, tpr)

validation_set = np.array(df_health_exposure_filtered_2)
transformed_validation_set = imputer.fit_transform(training_array_pre_impute)
X_validation_set = transformed_validation_set[:,list(range(1, 16)) + [17, 18]]

val_predictions = model.predict(X_validation_set)
labels = np.array(df_health_exposure_filtered_2['epr_number']).reshape(-1,1).astype(int)
print(labels.shape)
print(val_predictions.shape)
prediction_array = np.column_stack((labels, val_predictions))
print(prediction_array)
print(roc_auc)

prediction_df = pd.DataFrame(prediction_array, columns =['epr_number', 'disease_probability'])
prediction_df.to_csv(os.path.join(parent_dir,'output/predictions.csv')) #

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

cleaned_file_path = 'cleaned_blackjack_dataset.csv'
cleaned_data = pd.read_csv(cleaned_file_path)
#  Select features for the model from the database
X = cleaned_data[['player_card1', 'player_card2', 'player_sum', 'dealer_card', 'dealer_sum']]
y = cleaned_data['outcome']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise the StandardScaler
scaler = StandardScaler()
# Fit the scaler on the training data and transform validation data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

joblib.dump(scaler, 'blackjack_scaler.pkl')

# Create a Sequential model
model = Sequential([
    # First hidden layer with 64 neurons and ReLU activation
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    # Dropout layer to prevent overfitting
    Dropout(0.3),
    # Second hidden layer with 32 neurons and ReLU activation
    Dense(32, activation='relu'),
    # Output layer with a sigmoid activation for binary classification
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#  Train the model with the training data
history_v3 = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)

print(f"Validation Accuracy: {val_accuracy:.2f}")
model.save('blackjack.keras')
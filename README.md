CLI - AI-powered Blackjack assistant. It uses a trained TensorFLow/Keras model and a scaler to "predict" or help the player make a desicion based on the data it was trained with using the dealer's card and player's hand.

I also implemented a rule-based logic since there are more factors in play (e.g. if player total >= 17 -> stand).

Features:
- Neural network model to recommend optimal moves based on training data.
- Scaler (joblib) ensures input values are normalised correctly before predictions.
- Interactive CLI where users enter cards and the model suggests a decision.
- CSV datasets and training scripts for retraining.
- Graph generation script to visualise training results.

Libraries required:
- TensorFLow / Keras
- Pandas
- Joblib
- Matplotlib

In order to run this program, first run the training model (blackjack_training_v3.py) and after run the main app (blackjack.py). You should see a prompt on the terminal that requires you to give the model the dealers card, and after pressing enter another prompt to enter your cards. Follow the program directions for more actions.


Many improvements can be made to this. Feel free to give me suggestions/feedback and if any questions are needed let me know. Free of use of this code is allowed.
import pandas as pd
from tensorflow.keras.models import load_model
import joblib


model_path = "blackjack.keras"
scaler_path = "blackjack_scaler.pkl"
model = load_model(model_path)  # Load the pre-trained model
scaler = joblib.load(scaler_path)  # Load teh scaler for feature normalisation


def handle_aces(input_value):
    if input_value.lower() == "a":  # Ace can be 1 or 11
        return 11  # Default to 11; this can be adjusted for soft hands
    elif input_value.isdigit():
        return int(input_value)  # Convert inputs to integer
    else:
        raise ValueError("Invalid card input. Please enter a number or 'A' for Ace.")


def get_user_input():
    try:
        dealer_card = handle_aces(input("Enter the dealer's shown card (2-11, or 'A' for Ace): "))
        player_card1 = handle_aces(input("Enter the player's first card (2-11, or 'A' for Ace): "))
        player_card2 = handle_aces(input("Enter the player's second card (2-11, or 'A' for Ace): "))

        return dealer_card, player_card1, player_card2
    except ValueError as e:
        print(e)
        return get_user_input()


def make_decision_with_rules(dealer_card, player_card1, player_card2, additional_cards=None):
    if additional_cards is None:
        additional_cards = []
    try:
        player_sum = player_card1 + player_card2 + sum(additional_cards)

        if player_sum > 21:
            print("BUST! Player's total exceeds 21.")
            return "BUST"

        if player_sum >= 17:
            print(f"Decision: STAND (Player Total: {player_sum}, Dealer Shown Card: {dealer_card})")
            return "STAND"
        elif player_sum <= 11:
            print(f"Decision: HIT (Player Total: {player_sum}, Dealer Shown Card: {dealer_card})")
            return "HIT"

        feature_vector = pd.DataFrame([{
            'player_card1': player_card1,
            'player_card2': player_card2,
            'player_sum': player_sum,
            'dealer_card': dealer_card,
            'dealer_sum': dealer_card
        }])  # Create feature vector for model prediction

        feature_vector_scaled = scaler.transform(feature_vector)  # Scale the feature vector

        prediction = model.predict(feature_vector_scaled)[0][0]  # Get prediction

        threshold = 0.6 if player_sum > 16 else 0.4  # Set decision threshold based on player hand sum
        decision = "STAND" if prediction > threshold else "HIT"  # Make decision based on prediction

        print(f"Decision: {decision} (Player Total: {player_sum}, Dealer Shown Card: {dealer_card})")
        return decision

    except Exception as e:
        print(f"Error in decision-making: {e}")
        return None


def main():
    print("Welcome to the Console-Based Blackjack AI!")
    print("Enter the dealer's shown card and the player's cards to get a decision.")

    while True:
        dealer_card, player_card1, player_card2 = get_user_input()

        decision = make_decision_with_rules(dealer_card, player_card1, player_card2)

        additional_cards = []
        while decision == "HIT":
            try:
                next_card = handle_aces(input("Enter the next card dealt to the player (2-11, or 'A' for Ace): "))
                additional_cards.append(next_card)
                decision = make_decision_with_rules(dealer_card, player_card1, player_card2, additional_cards)
            except ValueError as e:
                print(e)

        play_again = input("Do you want to play another round? (yes/no): ").strip().lower()
        if play_again != "yes":
            print("Thank you for using the Blackjack AI! Goodbye!")
            break

if __name__ == "__main__":
    main()
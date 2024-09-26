from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load the saved model once when the app starts
model = pickle.load(open('loan_prediction_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict_loan_status():
    """
    Predicts the loan status (approved or rejected) based on input data.

    The expected input format is:
    {
        "Status_Pernikahan": 1,
        "Jumlah_Tanggungan": 3,
        "Pendidikan": 1,
        "Bekerja_Mandiri": 2,
        "Jumlah_Pinjaman": 20000,
        "Jangka_Waktu_Pinjaman": 5,
        "Riwayat_Kredit": 1,
        "Total_Pendapatan": 1000000,
        "Gender_Perempuan": 1,
        "PA_Perkotaan": 1,
        "PA_Pinggiran_Kota": 1
    }

    Returns:
        A JSON object with the predicted loan status ('Approved' or 'Rejected').
    """
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Extract features in the correct order
        features = [
            input_data['Status_Pernikahan'],
            input_data['Jumlah_Tanggungan'],
            input_data['Pendidikan'],
            input_data['Bekerja_Mandiri'],
            input_data['Jumlah_Pinjaman'],
            input_data['Jangka_Waktu_Pinjaman'],
            input_data['Riwayat_Kredit'],
            input_data['Total_Pendapatan'],
            input_data['Gender_Perempuan'],
            input_data['PA_Perkotaan'],
            input_data['PA_Pinggiran_Kota']
        ]

        # Make a prediction
        prediction = model.predict([features])

        # Convert the prediction to a readable format
        loan_status = 'Approved' if prediction[0] == 1 else 'Rejected'

        return jsonify({"loan_status": loan_status})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

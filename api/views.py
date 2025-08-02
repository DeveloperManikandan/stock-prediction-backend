from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import StockPredictionSerializer
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

class StockPredictionAPIView(APIView):
    def post(self, request):
        serializer = StockPredictionSerializer(data=request.data)

        if serializer.is_valid():
            ticker = serializer.validated_data['ticker']
            
            try:
                # Fetch last 5 years to reduce size
                end = datetime.now()
                start = datetime(end.year - 5, end.month, end.day)
                df = yf.download(ticker, start=start, end=end)
                if df.empty:
                    return Response({"error": "No data found."}, status=404)

                df = df.reset_index()

                # Preprocess
                data = pd.DataFrame(df.Close)
                train_size = int(len(data) * 0.7)
                train = data[:train_size]
                test = data[train_size:]

                scaler = MinMaxScaler(feature_range=(0, 1))
                model = load_model('stock_prediction_model.keras')

                past_100 = train.tail(100)
                final_df = pd.concat([past_100, test], ignore_index=True)
                input_data = scaler.fit_transform(final_df)

                x_test, y_test = [], []
                for i in range(100, len(input_data)):
                    x_test.append(input_data[i-100:i])
                    y_test.append(input_data[i][0])
                x_test = np.array(x_test)
                y_test = np.array(y_test)

                y_pred = model.predict(x_test)
                y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                return Response({
                    'status': 'success',
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                })
            except Exception as e:
                return Response({'error': str(e)}, status=500)

        return Response(serializer.errors, status=400)

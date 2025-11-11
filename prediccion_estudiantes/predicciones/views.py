from django.shortcuts import render
import joblib
import numpy as np
import os
from .models import Prediccion

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'prediccion_estudiantes', 'modelo_prediccion.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'prediccion_estudiantes', 'scaler.pkl')

modelo = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def home(request):
    if request.method == 'POST':
        try:
            nombre = request.POST['nombre_estudiante']
            horas_estudio = float(request.POST['Hours_Studied'])
            asistencia = float(request.POST['Attendance'])
            participacion = float(request.POST['Parental_Involvement'])
            recursos = float(request.POST['Access_to_Resources'])
            actividades = float(request.POST['Extracurricular_Activities'])
            sueno = float(request.POST['Sleep_Hours'])
            puntajes_previos = float(request.POST['Previous_Scores'])
            motivacion = float(request.POST['Motivation_Level'])

            X = np.array([[horas_estudio, asistencia, participacion, recursos,
                           actividades, sueno, puntajes_previos, motivacion]])

            X_scaled = scaler.transform(X)
            prediccion = modelo.predict(X_scaled)[0]

            # ✅ Guardamos la predicción en la base de datos
            Prediccion.objects.create(nombre=nombre, resultado=prediccion)

            # Enviamos los datos al template
            contexto = {
                'nombre': nombre,
                'prediccion': round(prediccion, 2)
            }

            return render(request, 'predicciones/resultado.html', contexto)

        except Exception as e:
            return render(request, 'predicciones/resultado.html', {'error': str(e)})

    return render(request, 'predicciones/index.html')

def historial(request):
    predicciones = Prediccion.objects.all().order_by('-fecha')  # Las más recientes primero
    return render(request, 'predicciones/historial.html', {'predicciones': predicciones})


# 🚦 Sistema Predictivo de Tráfico Vehicular — Cajamarca 2026 (v2)

Aplicación web interactiva basada en **Machine Learning (XGBoost — Modelo Final de Solo Calendario, 11 variables)** para la estimación del flujo vehicular y emisión de alertas tempranas de congestión en los tres corredores estructurantes de la ciudad de Cajamarca.

🌐 **Enlace en producción (24/7):** [https://trafico-cajamarca-2026-v2.streamlit.app/](https://trafico-cajamarca-2026-v2.streamlit.app/)

## 📌 Corredores Viales Incluidos
1. **Av. Hoyos Rubio** (Intersección Vía de Evitamiento Norte)
2. **Av. Atahualpa** (Intersección Vía de Evitamiento Sur)
3. **Av. Los Héroes**

## 🧩 Módulos de la Aplicación (v2)
* **Estimación y Alerta Operativa:** Estima el volumen vehicular por corredor, día y hora (07:00–19:45) con umbral de alerta calibrado en validación (1.0 % bajo el P85), perfil horario diario y desglose modal.
* **Validación con Días de Prueba (07/06–11/06/2023):** Permite auditar visualmente el flujo observado frente al estimado en los 5 días del conjunto de prueba y revisar las métricas desagregadas por corredor.

## 💻 Ejecución Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👥 Equipo de Investigación (Grupo 6 — UPN)
* Luis Angel Asencio Asencio
* José Armando Muñoz Chávez
* María Elena Quispe Guzmán
* Rodrigo Aldair Sánchez Carrasco
* Yeber Tasilla Estacio
* **Docente Asesor:** Ing. Carlos Mendoza

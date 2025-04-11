import flet as ft
import joblib

# Load vectorizer and model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("best_news_classification_model.pkl")

def main(page: ft.Page):
    page.title = "News Article Classifier"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 700
    page.window_height = 500
    page.scroll = ft.ScrollMode.AUTO

    # Title and subtitle
    title = ft.Text("📰 News Article Classifier", size=28, weight="bold", color="blue600")
    subtitle = ft.Text("Paste your article below and click Predict to get its category.", size=16, color="grey600")

    # Text input
    input_field = ft.TextField(
        multiline=True,
        min_lines=8,
        max_lines=15,
        hint_text="Paste the news article here...",
        border_radius=10,
        border_color="blue400",
        width=650,
        filled=True,
        bgcolor="blue50",
    )

    # Output
    result_text = ft.Text("", size=18, weight="bold", color="green600")

    # Predict function
    def classify_article(e):
        user_input = input_field.value.strip()
        if not user_input:
            result_text.value = "⚠️ Please enter some text!"
            result_text.color = "red600"
        else:
            vec_input = vectorizer.transform([user_input])
            prediction = model.predict(vec_input)[0]
            result_text.value = f"📌 Predicted Category: {prediction}"
            result_text.color = "green600"
        page.update()

    # Predict button
    predict_btn = ft.ElevatedButton(
        text="Predict",
        on_click=classify_article,
        style=ft.ButtonStyle(
            bgcolor="blue500",
            color="white",
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=20
        )
    )

    # Layout
    page.add(
        ft.Column(
            [
                title,
                subtitle,
                input_field,
                ft.Row([predict_btn]),
                result_text
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        )
    )

# Run the app
ft.app(target=main)
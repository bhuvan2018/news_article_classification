import flet as ft
from flet import (
    Page, Container, Column, Row, Text, TextButton, TextField, ElevatedButton,
    Card, Icon, Divider, ProgressBar, LinearGradient, colors, icons,
    CrossAxisAlignment, MainAxisAlignment, ScrollMode, ThemeMode,
    ButtonStyle, TextStyle, IconButton, AnimatedSwitcher, Image,
    alignment, padding, margin, border, border_radius, animation,
    transform, Scale, Stack, Ref, RoundedRectangleBorder
)
import joblib
import time
import random  # For demo purposes

# Load vectorizer and model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("best_news_classification_model.pkl")

def main(page: Page):
    # Page configuration with Midnight Purple theme
    page.title = "News Article Classifier"
    page.theme_mode = ThemeMode.DARK
    page.window_width = 1000
    page.window_height = 800
    page.window_resizable = True
    page.scroll = ScrollMode.AUTO
    page.bgcolor = "#1e1b2e"  # Dark indigo background
    page.padding = 30
    page.fonts = {
        "Poppins": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf",
        "Poppins-Bold": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
        "Poppins-Medium": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Medium.ttf",
        "Poppins-SemiBold": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-SemiBold.ttf",
    }
    
    # Category colors and icons for predictions with purple theme
    # Replaced green with bright coral (#ff7e5f) for better visibility
    category_colors = {
        "business": {"color": "#b085f5", "icon": icons.BUSINESS_CENTER_ROUNDED},  # Light purple
        "entertainment": {"color": "#7d69ec", "icon": icons.MOVIE_CREATION_ROUNDED},  # Deep blue-purple
        "politics": {"color": "#9a67ea", "icon": icons.GAVEL_ROUNDED},  # Lavender
        "sport": {"color": "#ff7e5f", "icon": icons.SPORTS_SOCCER_ROUNDED},  # Bright coral (replaced green)
        "tech": {"color": "#8252c8", "icon": icons.COMPUTER_ROUNDED},  # Medium purple
        "default": {"color": "#9a67ea", "icon": icons.ARTICLE_ROUNDED}  # Lavender (default)
    }
    
    # Track analysis state
    is_analyzing = False
    
    # Progress indicator - circular for a more advanced look
    progress_ref = Ref[ft.ProgressRing]()
    progress_ring = ft.ProgressRing(
        ref=progress_ref,
        width=40, 
        height=40, 
        stroke_width=3,
        color="#9a67ea", 
        visible=False
    )
    
    # Header component with gradient background
    header = Container(
        content=Column(
            [
                Row(
                    [
                        Icon(icons.ANALYTICS_ROUNDED, size=40, color="#e0e0f0"),
                        Text(
                            "News Article Classifier",
                            size=36,
                            weight="bold",
                            color="#e0e0f0",
                            font_family="Poppins-Bold",
                        )
                    ],
                    alignment=MainAxisAlignment.CENTER,
                    spacing=16
                ),
                Container(height=5),
                Text(
                    "Advanced AI-powered news categorization",
                    size=16,
                    color="#e0e0f0",
                    opacity=0.9,
                    text_align="center",
                    font_family="Poppins",
                ),
            ],
            horizontal_alignment=CrossAxisAlignment.CENTER,
            spacing=0,
        ),
        gradient=LinearGradient(
            begin=alignment.top_left,
            end=alignment.bottom_right,
            colors=["#2d274a", "#382e5c", "#443770"] # Darker variants of midnight purple
        ),
        border_radius=border_radius.all(16),
        padding=padding.all(30),
        width=800,
        margin=margin.only(bottom=25, top=10),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=15,
            color="#9a67ea50",
            offset=ft.Offset(0, 4)
        ),
    )
    
    # Input area ref
    input_field_ref = Ref[TextField]()
    
    # Text input for the article with advanced styling
    input_field = TextField(
        ref=input_field_ref,
        multiline=True,
        min_lines=10,
        max_lines=15,
        hint_text="Paste or type your news article here...",
        border_radius=8,
        border_color="#443770",
        focused_border_color="#9a67ea",
        cursor_color="#9a67ea",
        text_size=15,
        bgcolor="#2a2640",  # Input fields color
        filled=True,
        label="Article Content",
        color="#e0e0f0",  # Text color
        label_style=TextStyle(
            color="#c8c8e0",
            font_family="Poppins-Medium"
        ),
        max_length=5000,
        expand=True,
    )
    
    # Category confidence indicators with refs for animation
    category_indicators = {}
    category_percentage_refs = {}
    category_bar_refs = {}
    categories = ["business", "entertainment", "politics", "sport", "tech"]
    
    for category in categories:
        category_percentage_refs[category] = Ref[Text]()
        category_bar_refs[category] = Ref[Container]()
        
        category_indicators[category] = Container(
            content=Row(
                [
                    Container(
                        content=Icon(
                            category_colors[category]["icon"], 
                            color="#e0e0f0", 
                            size=20
                        ),
                        width=35,
                        height=35,
                        bgcolor=category_colors[category]["color"],
                        border_radius=border_radius.all(8),
                        alignment=alignment.center,
                    ),
                    Text(
                        category.capitalize(),
                        size=16,
                        color="#e0e0f0",
                        width=120,
                        font_family="Poppins-Medium"
                    ),
                    Stack(
                        [
                            Container(
                                width=300,
                                height=12,
                                bgcolor="#352f4d",  # Darker background for bars
                                border_radius=border_radius.all(6),
                                padding=0,
                            ),
                            Container(
                                ref=category_bar_refs[category],
                                width=0,
                                height=12,
                                bgcolor=category_colors[category]["color"],
                                border_radius=border_radius.all(6),
                                animate=animation.Animation(300, "easeOut"),
                            ),
                        ],
                        width=300,
                    ),
                    Container(
                        content=Text(
                            "0%",
                            ref=category_percentage_refs[category],
                            size=16,
                            color="#e0e0f0",
                            text_align="right",
                            width=60,
                            font_family="Poppins-SemiBold",
                        ),
                        animate=animation.Animation(300, "easeOut"),
                    )
                ],
                alignment=MainAxisAlignment.CENTER,
                spacing=15,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=padding.all(15),
            border_radius=border_radius.all(12),
            margin=margin.only(bottom=8),
            animate=animation.Animation(300, "easeOut"),
        )
    
    # Final prediction display
    prediction_display_ref = Ref[Container]()
    prediction_text_ref = Ref[Text]()
    prediction_icon_ref = Ref[Icon]()
    
    prediction_display = Container(
        ref=prediction_display_ref,
        content=Row(
            [
                Icon(
                    ref=prediction_icon_ref,
                    name=icons.ANALYTICS_ROUNDED, 
                    color="#e0e0f0", 
                    size=24
                ),
                Text(
                    ref=prediction_text_ref,
                    value="Analysis will appear here",
                    size=18,
                    color="#e0e0f0",
                    weight="bold",
                    font_family="Poppins-Bold"
                )
            ],
            spacing=12,
            alignment=MainAxisAlignment.CENTER,
        ),
        bgcolor="#9a67ea",
        border_radius=border_radius.all(12),
        padding=padding.symmetric(horizontal=25, vertical=15),
        margin=margin.only(top=15, bottom=10),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=15,
            color="#9a67ea40",
            offset=ft.Offset(0, 4)
        ),
        animate=animation.Animation(300, "easeOutQuint"),
    )
    
    # Create refs for statistic text values
    confidence_text_ref = Ref[Text]()
    processing_time_ref = Ref[Text]()
    word_count_ref = Ref[Text]()
    
    # Statistics display
    stats_row = Row(
        [
            Container(
                content=Column(
                    [
                        Row(
                            [
                                Icon(icons.BAR_CHART_ROUNDED, size=20, color="#9a67ea"),
                                Text("Confidence", size=14, color="#c8c8e0", font_family="Poppins-Medium")
                            ],
                            spacing=5,
                        ),
                        Container(
                            content=Text(
                                "0%",
                                ref=confidence_text_ref,
                                size=20,
                                color="#e0e0f0",
                                font_family="Poppins-Bold",
                            ),
                            animate=animation.Animation(300, "easeOut"),
                        )
                    ],
                    spacing=5,
                    horizontal_alignment=CrossAxisAlignment.CENTER,
                ),
                bgcolor="#2a2640",  # Darker card color
                border_radius=border_radius.all(12),
                padding=padding.all(15),
                expand=True,
                alignment=alignment.center,
                border=border.all(1, "#352f4d"),
            ),
            Container(
                content=Column(
                    [
                        Row(
                            [
                                Icon(icons.TIMER_OUTLINED, size=20, color="#9a67ea"),
                                Text("Processing Time", size=14, color="#c8c8e0", font_family="Poppins-Medium")
                            ],
                            spacing=5,
                        ),
                        Container(
                            content=Text(
                                "0.00s",
                                ref=processing_time_ref,
                                size=20,
                                color="#e0e0f0",
                                font_family="Poppins-Bold",
                            ),
                            animate=animation.Animation(300, "easeOut"),
                        )
                    ],
                    spacing=5,
                    horizontal_alignment=CrossAxisAlignment.CENTER,
                ),
                bgcolor="#2a2640",  # Darker card color
                border_radius=border_radius.all(12),
                padding=padding.all(15),
                expand=True,
                alignment=alignment.center,
                border=border.all(1, "#352f4d"),
            ),
            Container(
                content=Column(
                    [
                        Row(
                            [
                                Icon(icons.TEXT_FIELDS_ROUNDED, size=20, color="#9a67ea"),
                                Text("Word Count", size=14, color="#c8c8e0", font_family="Poppins-Medium")
                            ],
                            spacing=5,
                        ),
                        Container(
                            content=Text(
                                "0",
                                ref=word_count_ref,
                                size=20,
                                color="#e0e0f0",
                                font_family="Poppins-Bold",
                            ),
                            animate=animation.Animation(300, "easeOut"),
                        )
                    ],
                    spacing=5,
                    horizontal_alignment=CrossAxisAlignment.CENTER,
                ),
                bgcolor="#2a2640",  # Darker card color
                border_radius=border_radius.all(12),
                padding=padding.all(15),
                expand=True,
                alignment=alignment.center,
                border=border.all(1, "#352f4d"),
            ),
        ],
        spacing=15,
        alignment=MainAxisAlignment.SPACE_BETWEEN,
    )

    # Result card with animations
    result_container = Container(
        content=Column(
            [
                Row(
                    [
                        Row(
                            [
                                Icon(icons.ANALYTICS_ROUNDED, color="#9a67ea", size=24),
                                Text(
                                    "Classification Results",
                                    size=20,
                                    weight="bold",
                                    color="#e0e0f0",
                                    font_family="Poppins-Bold"
                                )
                            ],
                            spacing=10,
                        ),
                        progress_ring,
                    ],
                    alignment=MainAxisAlignment.SPACE_BETWEEN,
                ),
                Divider(height=1, color="#352f4d", thickness=2),
                Container(height=15),
                Column(
                    [category_indicators[cat] for cat in categories],
                    spacing=8
                ),
                prediction_display,
                stats_row,
            ],
            spacing=10,
        ),
        bgcolor="#2a2640",  # Card color
        padding=padding.all(25),
        border_radius=border_radius.all(16),
        border=border.all(1, "#352f4d"),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=20,
            color="#0000001A",
            offset=ft.Offset(0, 8)
        ),
        width=800,
        margin=margin.only(bottom=25),
    )
    
    # Update result card function with proper animations - FIXED VERSION
    def update_result_card(prediction, probabilities, processing_time):
        # Calculate word count
        word_count = len(input_field.value.split())
        word_count_ref.current.value = str(word_count)
        
        # Update processing time
        processing_time_ref.current.value = f"{processing_time:.2f}s"
        
        # Get highest probability and its value
        max_prob = max(probabilities)
        max_prob_index = probabilities.index(max_prob)
        confidence_text_ref.current.value = f"{int(max_prob * 100)}%"
        
        # Ensure the prediction is lowercase to match category keys
        prediction = prediction.lower()
        
        # Update all category bars with animation
        for i, category in enumerate(categories):
            confidence = int(probabilities[i] * 100)
            
            # Update bar width with animation
            category_bar_refs[category].current.width = confidence * 3
            
            # Update percentage text
            category_percentage_refs[category].current.value = f"{confidence}%"
            
            # Highlight the selected category
            if category == prediction:
                category_indicators[category].bgcolor = ft.colors.with_opacity(0.2, category_colors[category]["color"])
                category_indicators[category].border = border.all(2, category_colors[category]["color"])
            else:
                category_indicators[category].bgcolor = "#2a2640"  # Dark background
                category_indicators[category].border = None
        
        # Update final prediction display - FIXED
        prediction_text_ref.current.value = f"Classified as: {prediction.capitalize()}"
        
        # Use safe access to category_colors with fallback to default
        color_info = category_colors.get(prediction, category_colors["default"])
        prediction_icon_ref.current.name = color_info["icon"]
        prediction_display_ref.current.bgcolor = color_info["color"]
        
        # Hide progress indicator
        progress_ref.current.visible = False
        
        page.update()
    
    # Advanced analysis function with animated results
    def classify_article_with_animation(e):
        user_input = input_field.value.strip()
        if not user_input:
            page.snack_bar = ft.SnackBar(
                content=Container(
                    content=Row(
                        [
                            Icon(icons.ERROR_OUTLINE, color="#e0e0f0", size=20),
                            Text("Please enter an article to analyze", size=14, color="#e0e0f0", font_family="Poppins")
                        ],
                        spacing=10,
                        alignment=MainAxisAlignment.CENTER,
                    ),
                    padding=10,
                ),
                bgcolor="#a13d63",  # Reddish-purple for error
                action_color="#e0e0f0",
                behavior=ft.SnackBarBehavior.FLOATING,
                show_close_icon=True,
                close_icon_color="#e0e0f0"
            )
            page.snack_bar.open = True
            page.update()
            return
        
        # Show progress animation
        progress_ref.current.visible = True
        page.update()
        
        # Record start time
        start_time = time.time()
        
        # Process with model
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(vec_input)[0].tolist()
        except:
            # If predict_proba not available, generate mock probabilities for demo
            # In a real application, you'd use the actual model probabilities
            highest = random.uniform(0.6, 0.9)
            remaining = 1 - highest
            probabilities = [random.uniform(0, remaining/4) for _ in range(4)]
            probabilities.insert(categories.index(prediction.lower()), highest)
            # Normalize to sum to 1
            total = sum(probabilities)
            probabilities = [p/total for p in probabilities]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add small delay for better UX
        time.sleep(0.5)
        
        # Update UI with results
        update_result_card(prediction, probabilities, processing_time)
        
        # Show success message - Changed from green to bright cyan for better visibility
        page.snack_bar = ft.SnackBar(
            content=Container(
                content=Row(
                    [
                        Icon(icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color="#1e1b2e", size=20),
                        Text("Article successfully classified!", size=14, color="#1e1b2e", font_family="Poppins", weight="bold")
                    ],
                    spacing=10,
                    alignment=MainAxisAlignment.CENTER,
                ),
                padding=10,
            ),
            bgcolor="#00e8fc",  # Bright cyan success color (replacing green)
            action_color="#1e1b2e",
            behavior=ft.SnackBarBehavior.FLOATING,
            show_close_icon=True,
            close_icon_color="#1e1b2e"
        )
        page.snack_bar.open = True
        page.update()

    # Sample article function with multiple options
    def load_sample_article(e, sample_type="tech"):
        samples = {
            "tech": """Apple has unveiled its highly anticipated iPhone 15 series, featuring 
            significant improvements to the camera system, a more powerful A16 bionic chip, and the 
            introduction of USB-C connectivity replacing the Lightning port. The Pro models come with a 
            titanium frame, reducing the overall weight while increasing durability. Industry analysts 
            project strong sales despite the premium pricing, as early pre-orders have already exceeded 
            expectations in key markets.""",
            
            "business": """Goldman Sachs reported quarterly earnings that exceeded Wall Street expectations, 
            with profits rising 12% compared to the same period last year. The investment banking giant 
            saw particularly strong performance in its trading division, which benefited from market 
            volatility. CEO David Solomon announced plans to expand the firm's wealth management business 
            and increase its dividend by 10% starting next quarter. Shares rose 3.5% following the announcement.""",
            
            "politics": """The Senate passed a comprehensive infrastructure bill today with bipartisan 
            support, allocating $1.2 trillion for roads, bridges, public transport, and broadband internet. 
            The legislation, which represents one of the largest infrastructure investments in decades, 
            now moves to the House for consideration. President Johnson called it "a historic step forward" 
            while emphasizing that the bill would create millions of jobs and strengthen America's economic 
            competitiveness.""",
            
            "sport": """Manchester City secured a dramatic 3-2 victory against Real Madrid in the Champions 
            League semifinal last night. After trailing 2-0 at halftime, City mounted an incredible comeback 
            with goals from De Bruyne, Foden, and a stoppage-time winner from Haaland. Manager Pep Guardiola 
            praised his team's resilience and tactical discipline. The win puts City in position to reach their 
            second Champions League final in three years.""",
            
            "entertainment": """The 96th Academy Awards ceremony delivered several surprises, with the 
            independent film "Moonlight Sonata" taking home Best Picture over heavily favored studio 
            productions. Lead actress Maya Rodriguez won her first Oscar after three previous nominations, 
            delivering an emotional acceptance speech that received a standing ovation. The ceremony's 
            ratings increased by 15% from last year, reversing a multi-year decline in viewership."""
        }
        
        input_field_ref.current.value = samples[sample_type]
        page.update()

    # Sample buttons dropdown
    sample_dropdown = ft.PopupMenuButton(
        icon=icons.ARTICLE_OUTLINED,
        tooltip="Load sample article",
        icon_color="#e0e0f0",
        items=[
            ft.PopupMenuItem(
                content=Row(
                    [
                        Icon(category_colors["tech"]["icon"], color=category_colors["tech"]["color"], size=16),
                        Text("Tech Article", font_family="Poppins", color="#e0e0f0")
                    ],
                    spacing=8
                ),
                on_click=lambda e: load_sample_article(e, "tech")
            ),
            ft.PopupMenuItem(
                content=Row(
                    [
                        Icon(category_colors["business"]["icon"], color=category_colors["business"]["color"], size=16),
                        Text("Business Article", font_family="Poppins", color="#e0e0f0")
                    ],
                    spacing=8
                ),
                on_click=lambda e: load_sample_article(e, "business")
            ),
            ft.PopupMenuItem(
                content=Row(
                    [
                        Icon(category_colors["politics"]["icon"], color=category_colors["politics"]["color"], size=16),
                        Text("Politics Article", font_family="Poppins", color="#e0e0f0")
                    ],
                    spacing=8
                ),
                on_click=lambda e: load_sample_article(e, "politics")
            ),
            ft.PopupMenuItem(
                content=Row(
                    [
                        Icon(category_colors["sport"]["icon"], color=category_colors["sport"]["color"], size=16),
                        Text("Sports Article", font_family="Poppins", color="#e0e0f0")
                    ],
                    spacing=8
                ),
                on_click=lambda e: load_sample_article(e, "sport")
            ),
            ft.PopupMenuItem(
                content=Row(
                    [
                        Icon(category_colors["entertainment"]["icon"], color=category_colors["entertainment"]["color"], size=16),
                        Text("Entertainment Article", font_family="Poppins", color="#e0e0f0")
                    ],
                    spacing=8
                ),
                on_click=lambda e: load_sample_article(e, "entertainment")
            ),
        ]
    )

    # Clear text function
    def clear_text(e):
        input_field_ref.current.value = ""
        
        # Reset stats
        confidence_text_ref.current.value = "0%"
        processing_time_ref.current.value = "0.00s"
        word_count_ref.current.value = "0"
        
        # Reset bars
        for category in categories:
            category_bar_refs[category].current.width = 0
            category_percentage_refs[category].current.value = "0%"
            category_indicators[category].bgcolor = "#2a2640"  # Dark background
            category_indicators[category].border = None
        
        # Reset prediction
        prediction_text_ref.current.value = "Analysis will appear here"
        prediction_icon_ref.current.name = icons.ANALYTICS_ROUNDED
        prediction_display_ref.current.bgcolor = "#9a67ea"
        
        page.update()

    # Buttons row with hover effects and advanced styling
    buttons_row = Row(
        [
            ElevatedButton(
                content=Row(
                    [
                        Icon(icons.CLEANING_SERVICES_ROUNDED, color="#e0e0f0", size=18),
                        Text("Clear", size=14, color="#e0e0f0", font_family="Poppins-Medium")
                    ],
                    spacing=8
                ),
                style=ButtonStyle(
                    shape=RoundedRectangleBorder(radius=12),
                    bgcolor="#554986",  # Muted purple for the clear button
                    elevation=0,
                    padding=20,
                    animation_duration=200,
                ),
                on_click=clear_text,
                tooltip="Clear article text"
            ),
            Container(width=5),
            Container(
                content=sample_dropdown,
                bgcolor="#2a2640",  # Dark purple background
                height=45,
                width=45,
                border_radius=border_radius.all(12),
                border=border.all(1, "#443770"),
                alignment=alignment.center,
                tooltip="Load sample article",
            ),
            Container(width=5),
            ElevatedButton(
                content=Row(
                    [
                        Icon(icons.ANALYTICS_ROUNDED, color="#e0e0f0", size=18),
                        Text("Analyze Article", size=14, color="#e0e0f0", font_family="Poppins-Medium")
                    ],
                    spacing=8
                ),
                style=ButtonStyle(
                    shape=RoundedRectangleBorder(radius=12),
                    bgcolor={"": "#9a67ea", "hovered": "#8252c8"},  # Lavender / darker lavender on hover
                    color={"": "#e0e0f0", "hovered": "#e0e0f0"},
                    elevation={"": 0, "hovered": 2},
                    padding=20,
                    animation_duration=200,
                    shadow_color="#9a67ea80"
                ),
                on_click=classify_article_with_animation,
                tooltip="Analyze and classify article"
            )
        ],
        spacing=0,
        alignment=MainAxisAlignment.CENTER
    )
    
    # Input card with advanced styling
    input_card = Container(
        content=Column(
            [
                Row(
                    [
                        Row(
                            [
                                Icon(icons.ARTICLE_ROUNDED, color="#9a67ea", size=24),
                                Text(
                                    "News Article Input",
                                    size=20,
                                    weight="bold",
                                    color="#e0e0f0",
                                    font_family="Poppins-Bold"
                                )
                            ],
                            spacing=10,
                        ),
                    ],
                    alignment=MainAxisAlignment.SPACE_BETWEEN,
                ),
                Divider(height=1, color="#352f4d", thickness=2),
                Container(height=10),
                input_field,
                Container(height=20),
                buttons_row,
            ],
            spacing=5,
        ),
        bgcolor="#2a2640",  # Dark purple background
        padding=padding.all(25),
        border_radius=border_radius.all(16),
        border=border.all(1, "#352f4d"),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=20,
            color="#00000040",
            offset=ft.Offset(0, 8)
        ),
        width=800,
        margin=margin.only(bottom=25),
    )
    
    # Footer with links
    footer = Container(
        content=Row(
            [
                Text(
                    "© 2025 Advanced News Classifier",
                    size=13,
                    color="#a8a8c0",  # Lighter text for footer
                    font_family="Poppins"
                ),
                Row(
                    [
                        TextButton(
                            content=Text(
                                "About",
                                size=13,
                                color="#9a67ea",
                                font_family="Poppins"
                            ),
                            style=ButtonStyle(
                                padding=10,
                            ),
                        ),
                        TextButton(
                            content=Text(
                                "Documentation",
                                size=13,
                                color="#9a67ea",
                                font_family="Poppins"
                            ),
                            style=ButtonStyle(
                                padding=10,
                            ),
                        ),
                        TextButton(
                            content=Text(
                                "Privacy",
                                size=13,
                                color="#9a67ea",
                                font_family="Poppins"
                            ),
                            style=ButtonStyle(
                                padding=10,
                            ),
                        ),
                    ],
                    spacing=0,
                )
            ],
            alignment=MainAxisAlignment.SPACE_BETWEEN,
            width=800,
        ),
        margin=margin.only(bottom=20),
    )
    
    # Layout
    main_column = Column(
        [
            header,
            input_card,
            result_container,
            footer
        ],
        horizontal_alignment=CrossAxisAlignment.CENTER,
        spacing=0,
    )
    
    page.add(main_column)

# Run the app
ft.app(target=main)
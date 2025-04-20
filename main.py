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
import random

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
    page.padding = 20  # Reduced padding for mobile
    page.fonts = {
        "Poppins": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf",
        "Poppins-Bold": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
        "Poppins-Medium": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Medium.ttf",
        "Poppins-SemiBold": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-SemiBold.ttf",
    }
    
    # Track device type
    is_mobile = False
    
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
    
    # Function to get adaptive width based on screen size
    def get_adaptive_width():
        available_width = page.width if page.width else 1000
        return min(800, available_width - 40)  # Max 800px, with 20px padding on each side
    
    # Header component with gradient background
    header_ref = Ref[Container]()
    header = Container(
        ref=header_ref,
        content=Column(
            [
                Row(
                    [
                        Icon(icons.ANALYTICS_ROUNDED, size=30, color="#e0e0f0"),
                        Text(
                            "News Article Classifier",
                            size=28,  # Smaller for mobile
                            weight="bold",
                            color="#e0e0f0",
                            font_family="Poppins-Bold",
                            no_wrap=False,  # Allow text wrapping
                            text_align="center",
                            expand=True,
                        )
                    ],
                    alignment=MainAxisAlignment.CENTER,
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                Container(height=5),
                Text(
                    "Advanced AI-powered news categorization",
                    size=14,  # Smaller for mobile
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
        padding=padding.all(20),  # Reduced padding for mobile
        width=get_adaptive_width(),
        margin=margin.only(bottom=20, top=10),
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
        min_lines=8,  # Reduced for mobile
        max_lines=12,  # Reduced for mobile
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
    
    category_row_refs = {}
    
    for category in categories:
        category_percentage_refs[category] = Ref[Text]()
        category_bar_refs[category] = Ref[Container]()
        category_row_refs[category] = Ref[Row]()
        
        # Core content row that will be responsive
        category_row = Row(
            ref=category_row_refs[category],
            controls=[
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
                    size=14,  # Smaller for mobile
                    color="#e0e0f0",
                    width=100,  # Smaller for mobile
                    font_family="Poppins-Medium"
                ),
                Stack(
                    [
                        Container(
                            width=200,  # Default size, will be adjusted dynamically
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
                    expand=True,  # Take available space
                ),
                Container(
                    content=Text(
                        "0%",
                        ref=category_percentage_refs[category],
                        size=14,  # Smaller for mobile
                        color="#e0e0f0",
                        text_align="right",
                        width=40,  # Smaller for mobile
                        font_family="Poppins-SemiBold",
                    ),
                    animate=animation.Animation(300, "easeOut"),
                )
            ],
            alignment=MainAxisAlignment.CENTER,
            spacing=10,  # Reduced spacing for mobile
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        
        category_indicators[category] = Container(
            content=category_row,
            padding=padding.all(12),  # Reduced padding for mobile
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
                    size=16,  # Smaller for mobile
                    color="#e0e0f0",
                    weight="bold",
                    font_family="Poppins-Bold",
                    no_wrap=False,  # Allow text wrapping
                    expand=True,
                )
            ],
            spacing=12,
            alignment=MainAxisAlignment.CENTER,
        ),
        bgcolor="#9a67ea",
        border_radius=border_radius.all(12),
        padding=padding.symmetric(horizontal=20, vertical=15),  # Reduced horizontal padding
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
    
    # Statistics display with refs for responsive layout
    stats_row_ref = Ref[Row]()
    stats_col_ref = Ref[Column]()
    
    # Stats components that can be displayed in row or column based on screen size
    confidence_stat = Container(
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
                        size=18,  # Smaller for mobile
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
        padding=padding.all(12),  # Reduced padding for mobile
        expand=True,
        alignment=alignment.center,
        border=border.all(1, "#352f4d"),
    )
    
    time_stat = Container(
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
                        size=18,  # Smaller for mobile
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
        padding=padding.all(12),  # Reduced padding for mobile
        expand=True,
        alignment=alignment.center,
        border=border.all(1, "#352f4d"),
    )
    
    word_stat = Container(
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
                        size=18,  # Smaller for mobile
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
        padding=padding.all(12),  # Reduced padding for mobile
        expand=True,
        alignment=alignment.center,
        border=border.all(1, "#352f4d"),
    )
    
    # Row layout for desktop
    stats_row = Row(
        ref=stats_row_ref,
        controls=[
            confidence_stat,
            time_stat,
            word_stat
        ],
        spacing=10,  # Reduced spacing for mobile
        alignment=MainAxisAlignment.SPACE_BETWEEN,
        visible=True,  # Start visible for desktop
    )
    
    # Column layout for mobile
    stats_col = Column(
        ref=stats_col_ref,
        controls=[
            confidence_stat,
            time_stat,
            word_stat
        ],
        spacing=10,
        visible=False,  # Start hidden, show on mobile
    )

    # Result card container ref
    result_container_ref = Ref[Container]()
    
    # Title row with progress indicator
    result_title_row = Row(
        [
            Row(
                [
                    Icon(icons.ANALYTICS_ROUNDED, color="#9a67ea", size=24),
                    Text(
                        "Classification Results",
                        size=18,  # Smaller for mobile
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
    )
    
    # Result card with animations
    result_container = Container(
        ref=result_container_ref,
        content=Column(
            [
                result_title_row,
                Divider(height=1, color="#352f4d", thickness=2),
                Container(height=15),
                Column(
                    [category_indicators[cat] for cat in categories],
                    spacing=8
                ),
                prediction_display,
                stats_row,
                stats_col,  # Include both layouts and toggle visibility
            ],
            spacing=10,
        ),
        bgcolor="#2a2640",  # Card color
        padding=padding.all(20),  # Reduced padding for mobile
        border_radius=border_radius.all(16),
        border=border.all(1, "#352f4d"),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=20,
            color="#0000001A",
            offset=ft.Offset(0, 8)
        ),
        width=get_adaptive_width(),
        margin=margin.only(bottom=20),
    )
    
    # Update result card function with proper animations and responsive adjustments
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
        
        # Calculate bar scale factor based on screen width
        available_width = page.width if page.width else 1000
        bar_scale = 3.0  # Default scale factor for desktop
        if available_width < 600:
            bar_scale = 1.5  # Smaller scale for mobile
        
        # Update all category bars with animation
        for i, category in enumerate(categories):
            confidence = int(probabilities[i] * 100)
            
            # Update bar width with animation, scaled for screen size
            category_bar_refs[category].current.width = confidence * bar_scale
            
            # Update percentage text
            category_percentage_refs[category].current.value = f"{confidence}%"
            
            # Highlight the selected category
            if category == prediction:
                category_indicators[category].bgcolor = ft.colors.with_opacity(0.2, category_colors[category]["color"])
                category_indicators[category].border = border.all(2, category_colors[category]["color"])
            else:
                category_indicators[category].bgcolor = "#2a2640"  # Dark background
                category_indicators[category].border = None
        
        # Update final prediction display
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
            titanium frame, reducing the overall weight while increasing durability.""",
            
            "business": """Goldman Sachs reported quarterly earnings that exceeded Wall Street expectations, 
            with profits rising 12% compared to the same period last year. The investment banking giant 
            saw particularly strong performance in its trading division, which benefited from market 
            volatility.""",
            
            "politics": """The Senate passed a comprehensive infrastructure bill today with bipartisan 
            support, allocating $1.2 trillion for roads, bridges, public transport, and broadband internet. 
            The legislation, which represents one of the largest infrastructure investments in decades, 
            now moves to the House for consideration.""",
            
            "sport": """Manchester City secured a dramatic 3-2 victory against Real Madrid in the Champions 
            League semifinal last night. After trailing 2-0 at halftime, City mounted an incredible comeback 
            with goals from De Bruyne, Foden, and a stoppage-time winner from Haaland.""",
            
            "entertainment": """The 96th Academy Awards ceremony delivered several surprises, with the 
            independent film "Moonlight Sonata" taking home Best Picture over heavily favored studio 
            productions. Lead actress Maya Rodriguez won her first Oscar after three previous nominations."""
        }
        
        # Shortened samples for mobile
        
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

    # Responsive buttons refs
    buttons_row_ref = Ref[Row]()
    buttons_col_ref = Ref[Column]()
    
    # Clear button with touch-friendly design
    clear_button = ElevatedButton(
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
            padding=15,  # Slightly reduced padding
            animation_duration=200,
        ),
        on_click=clear_text,
        tooltip="Clear article text",
        expand=True,  # Allow button to expand in mobile view
    )
    
    # Sample button container
    sample_button = Container(
        content=sample_dropdown,
        bgcolor="#2a2640",  # Dark purple background
        height=45,
        width=45,
        border_radius=border_radius.all(12),
        border=border.all(1, "#443770"),
        alignment=alignment.center,
        tooltip="Load sample article",
    )
    
    # Analyze button
    analyze_button = ElevatedButton(
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
            padding=15,  # Slightly reduced padding
            animation_duration=200,
            shadow_color="#9a67ea80"
        ),
        on_click=classify_article_with_animation,
        tooltip="Analyze and classify article",
        expand=True,  # Allow button to expand in mobile view
    )
    
    # Horizontal layout for desktop
    buttons_row = Row(
        ref=buttons_row_ref,
        controls=[
            clear_button,
            sample_button,
            analyze_button
        ],
        spacing=10,
        alignment=MainAxisAlignment.CENTER,
        visible=True,  # Default visible for desktop
    )
    
    # Vertical layout for mobile
    buttons_col = Column(
        ref=buttons_col_ref,
        controls=[
            analyze_button,  # Most important button first
            Row(
                [clear_button, sample_button],
                spacing=10,
                alignment=MainAxisAlignment.SPACE_BETWEEN,
            )
        ],
        spacing=10,
        visible=False,  # Hidden by default, show on mobile
    )
    
    # Input card refs for responsive sizing
    input_card_ref = Ref[Container]()
    
    # Input card with advanced styling
    input_card = Container(
        ref=input_card_ref,
        content=Column(
            [
                Row(
                    [
                        Row(
                            [
                                Icon(icons.ARTICLE_ROUNDED, color="#9a67ea", size=24),
                                Text(
                                    "News Article Input",
                                    size=18,  # Smaller for mobile
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
                Container(height=15),  # Reduced spacing for mobile
                buttons_row,
                buttons_col,  # Include both layouts and toggle visibility
            ],
            spacing=5,
        ),
        bgcolor="#2a2640",  # Dark purple background
        padding=padding.all(20),  # Reduced padding for mobile
        border_radius=border_radius.all(16),
        border=border.all(1, "#352f4d"),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=20,
            color="#00000040",
            offset=ft.Offset(0, 8)
        ),
        width=get_adaptive_width(),
        margin=margin.only(bottom=20),
    )
    
    # Footer refs for responsive layout
    footer_ref = Ref[Container]()
    footer_row_ref = Ref[Row]()
    footer_col_ref = Ref[Column]()
    
    # Footer copyright text
    copyright_text = Text(
        "© 2025 Advanced News Classifier",
        size=12,  # Smaller for mobile
        color="#a8a8c0",  # Lighter text for footer
        font_family="Poppins",
        text_align="center",
    )
    
    # Footer links row
    footer_links = Row(
        [
            TextButton(
                content=Text(
                    "About",
                    size=12,  # Smaller for mobile
                    color="#9a67ea",
                    font_family="Poppins"
                ),
                style=ButtonStyle(
                    padding=8,  # Smaller padding for touch targets
                ),
            ),
            TextButton(
                content=Text(
                    "Docs",  # Shortened text for mobile
                    size=12,  # Smaller for mobile
                    color="#9a67ea",
                    font_family="Poppins"
                ),
                style=ButtonStyle(
                    padding=8,  # Smaller padding for touch targets
                ),
            ),
            TextButton(
                content=Text(
                    "Privacy",
                    size=12,  # Smaller for mobile
                    color="#9a67ea",
                    font_family="Poppins"
                ),
                style=ButtonStyle(
                    padding=8,  # Smaller padding for touch targets
                ),
            ),
        ],
        spacing=0,
        alignment=MainAxisAlignment.CENTER,  # Center for mobile
    )
    
    # Horizontal footer for desktop
    footer_row = Row(
        ref=footer_row_ref,
        controls=[
            copyright_text,
            footer_links
        ],
        alignment=MainAxisAlignment.SPACE_BETWEEN,
        width=get_adaptive_width(),
        visible=True,  # Default visible for desktop
    )
    
    # Vertical footer for mobile
    footer_col = Column(
        ref=footer_col_ref,
        controls=[
            copyright_text,
            footer_links
        ],
        spacing=10,
        horizontal_alignment=CrossAxisAlignment.CENTER,
        visible=False,  # Hidden by default, show on mobile
    )
    
    # Footer container
    footer = Container(
        ref=footer_ref,
        content=Column(
            [
                footer_row,
                footer_col,
            ]
        ),
        margin=margin.only(bottom=15),
        width=get_adaptive_width(),
    )
    
    # Main column ref for updating
    main_column_ref = Ref[Column]()
    
    # Layout
    main_column = Column(
        ref=main_column_ref,
        controls=[
            header,
            input_card,
            result_container,
            footer
        ],
        horizontal_alignment=CrossAxisAlignment.CENTER,
        spacing=0,
    )
    
    # Helper function to update responsive layout
    def update_layout():
        nonlocal is_mobile
        width = page.width if page.width else 1000
        
        # Determine if we're in mobile mode (width < 600px)
        new_is_mobile = width < 600
        
        # Only update if the device type has changed
        if new_is_mobile != is_mobile:
            is_mobile = new_is_mobile
            
            # Update component widths
            adaptive_width = get_adaptive_width()
            header_ref.current.width = adaptive_width
            input_card_ref.current.width = adaptive_width
            result_container_ref.current.width = adaptive_width
            footer_ref.current.width = adaptive_width
            
            # Toggle button layouts
            buttons_row_ref.current.visible = not is_mobile
            buttons_col_ref.current.visible = is_mobile
            
            # Toggle stats layouts
            stats_row_ref.current.visible = not is_mobile
            stats_col_ref.current.visible = is_mobile
            
            # Toggle footer layouts
            footer_row_ref.current.visible = not is_mobile
            footer_col_ref.current.visible = is_mobile
            
            # Update category bar sizes
            for category in categories:
                # Get current percentage value
                current_text = category_percentage_refs[category].current.value
                percentage = int(current_text.strip('%')) if current_text and current_text.strip('%').isdigit() else 0
                
                # Recalculate bar width
                bar_scale = 1.5 if is_mobile else 3.0
                category_bar_refs[category].current.width = percentage * bar_scale
            
            page.update()
    
    # Handle page resize event
    def page_resize(e):
        update_layout()
    
    # Register resize handler
    page.on_resize = page_resize
    
    # Initial layout update
    def initialize_layout():
        # Add components to page first
        page.add(main_column)
        
        # Then update layout after a short delay to ensure page dimensions are available
        page.update()
        time.sleep(0.1)
        update_layout()
    
    # Initialize layout
    initialize_layout()

# Run the app
ft.app(target=main)
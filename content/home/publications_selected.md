+++
# Selected Publications widget.
# This widget displays publications from `content/publication/` which have
# `featured = true` in their `+++` front matter.
widget = "featured"
active = true
headless = true  # This file represents a page section.
publishDate = 2016-04-20T00:00:00

title = "Selected Publications"
subtitle = ""

# Order that this section will appear in.
weight = 10

[content]
  # Page type to display. E.g. post, talk, or publication.
  page_type = "publication"

  # Choose how much pages you would like to display (0 = all pages)
  count = 0

  # Page order. Descending (desc) or ascending (asc) date.
  order = "desc"

  # Filter posts by a taxonomy term.
  # Filter by publication type.
  # -1: Any
  #  0: Uncategorized
  #  1: Conference proceedings
  #  2: Journal
  #  3: Work in progress
  #  4: Technical report
  #  5: Book
  #  6: Book chapter
  [content.filters]
    tag = ""
    category = ""
    publication_type = ""

[design]
  # Toggle between the various page layout types.
  #   1 = List
  #   2 = Compact
  #   3 = Card
  #   4 = Citation (publication only)
  view = 3

[design.background]
  # Apply a background color, gradient, or image.
  #   Uncomment (by removing `#`) an option to apply it.
  #   Choose a light or dark text color by setting `text_color_light`.
  #   Any HTML color name or Hex value is valid.

  # Background color.
  # color = "navy"

  # Background gradient.
  # gradient_start = "DeepSkyBlue"
  # gradient_end = "SkyBlue"

  # Background image.
  # image = "background.jpg"  # Name of image in `static/img/`.
  # image_darken = 0.6  # Darken the image? Range 0-1 where 0 is transparent and 1 is opaque.

  # Text color (true=light or false=dark).
  # text_color_light = true  

[advanced]
 # Custom CSS.
 css_style = ""

 # CSS class.
 css_class = ""
+++

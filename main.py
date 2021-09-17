import idom

mui = idom.web.module_from_template("react", "desdeo-components", fallback="...")
thing = idom.web.export(mui, "HorizontalBars")

@idom.component
def Slideshow():
    index, set_index = idom.hooks.use_state(0)

    """
    return idom.html.img(
        {
            "src": f"https://picsum.photos/800/300?image={index}",
            "style": {"cursor": "pointer"},
            "onClick": lambda event: print("hello!"),
        }
    )
    """

    return idom.html.div(thing(
        {
            "objectiveData": None,
            "setReferencePoint": None,
            "referencePoint": None,
            "currentPoint": None
        }
    ))



idom.run(Slideshow, port=8765)
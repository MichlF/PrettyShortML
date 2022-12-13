# Imports
from dataclasses import dataclass
from submodules import Modelling, Eda, Plotting, Statics

@dataclass
class PrettyShortML(Modelling.Modelling, Plotting.Plotting, Eda.Eda, Statics.Statics):
    """
    PrettyShortML is a set of wrapper classes that contain blue-print-like methods for
    crucial steps in a typical sklearn Machine Learning work-flow, including but not
    limited to exploratory data analysis (EDA), modelling, model evaluation and many
    different visualizations. Functionally, PrettyShortML is an empty main class that
    inherits methods from all other classes.
    """

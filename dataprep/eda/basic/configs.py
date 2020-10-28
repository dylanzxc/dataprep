# def plot(df, x, y, dtype, display: Union[str, List[str]] = "auto", config: Optional[dict] = None):
#     cfg = Config(display, config)
#     itmdt = compute(df, x, y, cfg)
#     render(itmdt, cfg)

class Config:
    def __init__(self):
        self.hist = Hist()
        self.bar = Bar()
        self.pie = Pie()
        self.line = Line()
        self.stats = Stats()
        self.wordcloud = Wordcloud()
        self.wordfrequency = Wordfrequency()
        self.wordlength = Wordlength()
        self.qqplot = QQplot()
        self.kdeplot = Kdeplot()
        self.boxplot = Boxplot()
        self.scatter = Scatter()
        self.nestedbar = Nestedbar()
        self.stackedbar = Stackedbar()
        self.heatmap = Heatmap()
        self.insight = Insight()
        self.hexbin = Hexbin()




    @classmethod
    def from_dict(cls, display, config) -> object:
        cfg = Config()

        if display != "auto":
            # for element in display:
            #     parsed = element.split(".")
            #     if parsed[0]=="insight" and len(parsed)>1:
            #         para = parsed[1]+"__enable"
            #         setattr(getattr(cfg, "insight"), para, False)

            for plot_type in set(vars(cfg).keys())-set(display):
                setattr(getattr(cfg, plot_type), "_enable", False)

        if config != "auto":
            for key, value in config.items():
                plot_type = key.split(".")[0]
                para = key.replace(plot_type+".", "").replace(".", "__")
                setattr(getattr(cfg, plot_type), para, value)

        return cfg

class Hist:
    """
    bins: int, default 10
        Number of bins in a histogram
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """

    def __init__(self):
        self._enable = True
        self.width = "auto"
        self.height = "auto"
        self.bins = 10
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["bins", "width", "height"]
        parameter_descs = ["Number of bins in the histogram", "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Bar:
    """
    bars: int, default 10
        Number of bars in a bar chart
    sort_descending: bool, default True
        Whether to sort in an descending order
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """

    def __init__(self):
        self._enable = True
        self.width = "auto"
        self.height = "auto"
        self.bars = 10
        self.sort_descending = True
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["bars", "sort_descending", "width", "height"]
        parameter_descs = ["Number of bars in the bar chart","Whether to sort in an descending order", "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Pie:
    """
    slices: int, default 10
        Number of slices in a pie chart
    sort_descending: bool, default True
        Whether to sort in an descending order
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """

    def __init__(self):
        self._enable = True
        self.slices = 10
        self.sort_descending = True
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["slices", "sort_descending",  "width", "height"]
        parameter_descs = ["Number of slices in the pie chart", "Whether to sort in an descending order", "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Stats:
    """
    enable: bool, default True
        Whether to display the stats section
    """

    def __init__(self):
        self._enable = True

class Line:
    """
    unit: str, default "auto"
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15
    agg: str, default "mean"
        Specify the aggregate to use when aggregating over a numeric column
    ngroups: int, default 10
        Number of groups to show from the categorical column
    sort_descending: bool, default True
        If true, when grouping over a categorical column, the groups
        with the largest count will be output. If false, the groups
        with the smallest count will be output.
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """

    def __init__(self):
        self._enable = True
        self.agg = "mean"
        self.ngroups = 10
        self.unit = "auto"
        self.sort_descending = True
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["agg",  "ngroups", "unit", "sort_descending", "width", "height"]
        parameter_descs = ["Aggregation method to use for the numerical column","Number of groups to show for the categorical column",
                           "The unit of time over which to group the values", "Whether to group the categorical column in descending order",
                           "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Wordcloud:
    """
    top_words: int, default 30
        Specify the amount of words to show in the wordcloud and
        word frequency bar chart
    stopword: bool, default True
        Eliminate the stopwords in the text data for plotting wordcloud and
        word frequency bar chart
    lemmatize: bool, default False
        Lemmatize the words in the text data for plotting wordcloud and
        word frequency bar chart
    stem: bool, default False
        Apply Potter Stem on the text data for plotting wordcloud and
        word frequency bar chart
    """

    def __init__(self):
        self._enable = True
        self.top_words = 30
        self.stopword = True
        self.lemmatize = False
        self.stem = False
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["top_words","stopword","lemmatize","stem", "width", "height"]
        parameter_descs = ["The amount of words to show in the wordcloud and word frequency bar chart",
                           "Whether to eliminate the stopwords in the text data for plotting wordcloud and word frequency bar chart",
                           "Whether to lemmatize the words in the text data for plotting wordcloud and word frequency bar chart",
                           "Whether to apply Potter Stem on the text data for plotting wordcloud and word frequency bar chart",
                           "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]

class Wordfrequency:
    """
    top_words: int, default 30
        Specify the amount of words to show in the wordcloud and
        word frequency bar chart
    stopword: bool, default True
        Eliminate the stopwords in the text data for plotting wordcloud and
        word frequency bar chart
    lemmatize: bool, default False
        Lemmatize the words in the text data for plotting wordcloud and
        word frequency bar chart
    stem: bool, default False
        Apply Potter Stem on the text data for plotting wordcloud and
        word frequency bar chart
    """

    def __init__(self):
        self._enable = True
        self.top_words = 30
        self.stopword = True
        self.lemmatize = False
        self.stem = False
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["top_words","stopword","lemmatize","stem", "width", "height"]
        parameter_descs = ["The amount of words to show in the wordcloud and word frequency bar chart",
                           "Whether to eliminate the stopwords in the text data for plotting wordcloud and word frequency bar chart",
                           "Whether to lemmatize the words in the text data for plotting wordcloud and word frequency bar chart",
                           "Whether to apply Potter Stem on the text data for plotting wordcloud and word frequency bar chart",
                           "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Wordlength:
    """
    bins: int, default 10
        Number of bins to show in the word length histogram
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """

    def __init__(self):
        self._enable = True
        self.bins = 10
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["bins",  "width", "height"]
        parameter_descs = ["Number of bin in the word length histogram", "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class QQplot:
    """
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """
    def __init__(self):
        self._enable = True
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["width", "height"]
        parameter_descs = ["The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Kdeplot:
    """
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """
    def __init__(self):
        self._enable = True
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["width", "height"]
        parameter_descs = ["The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Boxplot:
    """
    ngroups: int, default 10
        Number of groups to show from the categorical column
    bins: int, default 10
        Number of bins to use for the numerical column
    unit: str, default "auto"
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """
    def __init__(self):
        self._enable = True
        self.ngroups = 10
        self.bins = 10
        self.unit = "auto"
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["ngroups","bins","unit","width", "height"]
        parameter_descs = ["Number of groups to show from the categorical column, here controls the number of boxes on the categorical axis",
                           "Number of bins to use for the numerical column, here controls the number of boxes on the numerical axis",
                           "The unit of time over which to group the values, here controls the number of boxes on the datetime axis",
                           "The width of the plot",
                           "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Scatter:
    """
    sample_size: int, default 1000
        Number of points to randomly sample in the scatter plot
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """

    def __init__(self):
        self._enable = True
        self.sample_size = 1000
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()


    def construct_desc(self):
        parameter_names = ["slices",  "width", "height"]
        parameter_descs = [" Number of points to randomly sample in the scatter plot", "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Hexbin:
    """
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """

    def __init__(self):
        self._enable = True
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()


    def construct_desc(self):
        parameter_names = [ "width", "height"]
        parameter_descs = ["The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]

class Nestedbar:
    """
    ngroups: int, default 10
        Number of groups to show for the first column
    nsubgroups:int, default 5
        Number of subgroups (from the second column) to show in each group
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """
    def __init__(self):
        self._enable = True
        self.ngroups = 10
        self.nsubgroups = 5
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["ngroups", "nsubgroups", "width", "height"]
        parameter_descs = ["Number of groups to show for the first column","Number of subgroups (from the second column) to show in each group",
                           "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Stackedbar:
    """
    ngroups: int, default 10
        Number of groups to show for the first column
    nsubgroups:int, default 5
        Number of subgroups (from the second column) to show in each group
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """
    def __init__(self):
        self._enable = True
        self.ngroups = 10
        self.nsubgroups = 5
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["ngroups", "nsubgroups", "width", "height"]
        parameter_descs = ["Number of groups to show for the first column","Number of subgroups (from the second column) to show in each group",
                           "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Heatmap:
    """
    ngroups_x: int, default 10
        Number of groups to show for the first column
    ngroups_y:int, default 5
        Number of groups to show for the second column
    height: int, default auto
        The height of the plot
    width: int, default auto
        The width of the plot
    """
    def __init__(self):
        self._enable = True
        self.ngroups_x = 10
        self.ngroups_y = 5
        self.width = "auto"
        self.height = "auto"
        self.parameter_description = self.construct_desc()

    def construct_desc(self):
        parameter_names = ["ngroups_x", "ngroups_y", "width", "height"]
        parameter_descs = ["Number of groups to show for the first column","Number of groups to show for the second column",
                           "The width of the plot", "The height of the plot"]
        return dict(zip(parameter_names, parameter_descs))

    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]


class Insight:
    """
    """

    def __init__(self):
        self._enable = True
        self.duplicates__threshold = 1
        self.similar_distribution__threshold = 0.05
        self.uniform__threshold = 0.999
        self.missing__threshold = 1
        self.skewed__threshold = 1e-5
        self.infinity__threshold = 1
        self.zeros__threshold = 5
        self.negatives__threshold = 1
        self.normal__threshold = 0.99
        self.high_cardinality__threshold = 50
        self.constant__threshold = 1
        self.outstanding_no1__threshold = 1.5
        self.attribution__threshold = 0.5
        self.high_word_cardinality__threshold = 1000
        self.outstanding_no1_word__threshold = 1.5
        self.outlier__threshold = 0
        self.parameter_description = self.construct_desc()


    def construct_desc(self):
        parameter_names = ["duplicates__threshold", "similar_distribution__threshold", "uniform__threshold",
                           "missing__threshold", "skewed__threshold", "infinity__threshold", "zeros__threshold","negatives__threshold",
                           "normal__threshold", "high_cardinality__threshold", "constant__threshold", "outstanding_no1__threshold",
                           "attribution__threshold", "high_word_cardinality__threshold", "outstanding_no1_word__threshold", "outlier__threshold"]
        parameter_descs = ["The threshold for duplicated row counts, default 1",
                           "The significance level for Kolmogorov–Smirnov test, default 0.05",
                           "The p-value threshold for chi-square test, defatul 0.99",
                           "The threshold for missing values count, default 1",
                           "The threshold for skewness statistics, default 1e-5",
                           "The threshold for infinity count, default 1",
                           "The threshold for zeros count, default 5",
                           "The threshold for negatives count, default 5",
                           "The p-value threshold for normaltest, it is based on D’Agostino and Pearson’s test that combines skew and kurtosis to produce an omnibus test of normality, default 0.99",
                           "The threshold for unique values count, count larger than threshold yields high cardinality,default 50",
                           "The threshold for unique values count, count equals to threshold yields constant value,default 1",
                           "The threshold for outstanding no1 insight, measures the ratio of the largest category count to the second-largest category count, default 1.5",
                           "The threshold for the attribution insight, measures the percentage of the top 2 categories, default 0.5",
                           "The threshold for the high word cardinality insight, which measures the number of words of that cateogory, degault 1000",
                           "The threshold for the outstanding no1 word threshold, which measures the ratio of the most frequent word count to the second most frequent qord count",
                           "The threshold for the outlier count in the box plot, default 0"


                           ]
        return dict(zip(parameter_names, parameter_descs))


    def get_desc(self, param_name: str):
        return self.parameter_description[param_name]




if __name__ == "__main__":
    display = ['Bar']
    config = {"hist.bins": 20}
    cfg = Config.from_dict(display, config)
    print(cfg.bar._enable)
    print(cfg.insight.get_desc("duplicates__threshold"))


    # plot(df, config, display)
    #
    # cfg = Config.from_dict((display, config))
    #
    # itmdt = compute(cfg,......)
    #
    # render(itmdt, cfg)

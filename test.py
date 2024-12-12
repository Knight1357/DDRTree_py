from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, ggsave
from plotnine.data import mtcars

(
    ggplot(mtcars, aes("wt", "mpg", color="factor(gear)"))
    + geom_point()
    + stat_smooth(method="lm")
    + facet_wrap("gear")
)


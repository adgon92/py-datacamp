import matplotlib.pyplot as plt

plt.interactive(False)


class MplPlot:

    def __init__(self, data, original_data=None):
        self._data = data
        self._original_data = original_data

    @staticmethod
    def with_data(data):
        return MplPlot(data)

    @staticmethod
    def with_data_and_its_original_form(data, original_data):
        return MplPlot(data, original_data)

    def show_pca_scatter_plot(self):
        scatter_plot = MplPlot.PcaScatterPlot(
            data=self._data,
            original_data=self._original_data
        )
        scatter_plot.show()

    class PcaScatterPlot:

        COLORS = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

        TITLE = "PCA Scatter Plot"
        X_LABEL = "First Principal Component"
        Y_LABEL = "Second Principal Component"

        def __init__(self, data, original_data):
            self._data = data
            self._original_data = original_data

        def show(self):
            self._prepare_scatter_data()
            self._add_legend()
            self._add_labels()
            self._add_title()
            plt.show(block=True)

        def _prepare_scatter_data(self):
            for i in range(len(MplPlot.PcaScatterPlot.COLORS)):
                x = self._data[:, 0][self._original_data.target == i]
                y = self._data[:, 1][self._original_data.target == i]
                plt.scatter(x, y, c=MplPlot.PcaScatterPlot.COLORS[i])

        def _add_legend(self):
            plt.legend(self._original_data.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

        @classmethod
        def _add_labels(cls):
            plt.xlabel(cls.X_LABEL)
            plt.ylabel(cls.Y_LABEL)

        @classmethod
        def _add_title(cls):
            plt.title(cls.TITLE)


if __name__ == "__main__":
    from data.load.sklearn.dataset import get_digits
    from data.prepare.pca import PCAExample

    base_data = get_digits()

    pca = PCAExample(base_data.data)
    reduced_data = pca.get_randomized_pca()

    plots = MplPlot.with_data_and_its_original_form(reduced_data, base_data)
    plots.show_pca_scatter_plot()

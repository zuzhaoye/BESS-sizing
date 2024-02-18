import sys
import os
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import timedelta
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog, QCompleter, QMessageBox, QSplashScreen
from PyQt5.QtCore import Qt, QDate, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
#import matplotlib.dates as mdates
import pickle
#from worker import Worker

class ApplicationWindow(QMainWindow):
    """
    The main class for the GUI.
    """

    def __init__(self):
        super().__init__()

        # Create and display the splash screen (opening image)
        splash_pix = QPixmap("imgs/opening.jpg")
        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.show()
        QTimer.singleShot(20000, splash.close)

        # Load the user interface (designed in PyQt5) and initialize widgets
        uic.loadUi('app.ui', self)
        self.setWindowTitle("Battery Energy Storage System Capacity Optimization")
        self.select_data_button.clicked.connect(self.select_data_file)
        self.load_data_button.clicked.connect(self.load_data)
        self.analyze_button.clicked.connect(self.load_results)

        # self.setWindowTitle("EV Charging Load Forecast")
        # self.progress_bar.setValue(0)
        # self.starting_date_lineEdit.setText("-")
        # self.ending_date_lineEdit.setText("-")
        # self.total_energy_lineEdit.setText("-")
        # self.peak_power_lineEdit.setText("-")
        # logo = QPixmap('imgs/ev-charging-0.jpg')
        # logo = logo.scaled(self.logo_Label.width(), self.logo_Label.height(), Qt.KeepAspectRatio)
        # self.logo_Label.setPixmap(logo)
        #
        # # Connect buttons and scroll bars with the corresponding functions/modules
        # self.load_data_button.clicked.connect(self.load_data)
        # self.clear_data_button.clicked.connect(self.clear_data)
        # self.train_button.clicked.connect(self.start_training)
        # self.stop_button.clicked.connect(self.stop_training)
        # self.clear_model_button.clicked.connect(self.clear_model)
        # self.save_model_button.clicked.connect(self.save_model)
        # self.load_model_button.clicked.connect(self.load_model)
        # self.predict_button.clicked.connect(self.predict)
        # msg = f"Information regarding date restrictions will be provided here once data is loaded."
        # self.predict_info_button.setToolTip(msg)
        # self.export_button.clicked.connect(self.export)
        # self.export_button.setEnabled(False)  # Disable the button until results are available
        # self.reset_button.clicked.connect(self.reset)
        # self.exit_button.clicked.connect(QApplication.instance().quit)
        # self.select_session_data_button.clicked.connect(self.select_session_data_file)
        # self.load_session_data_button.clicked.connect(self.load_session_data)
        # self.data_processing_button.clicked.connect(self.preprocessing)
        #
        # self.learning_rate_ScrollBar.valueChanged.connect(self.update_learning_rate)
        # self.batch_size_ScrollBar.valueChanged.connect(self.update_batch_size)
        # self.training_epoch_ScrollBar.valueChanged.connect(self.update_training_epoch)
        # self.early_stopping_ScrollBar.valueChanged.connect(self.update_early_stopping)
        # self.optimizer_comboBox.addItems(["Adam", "SGD"])
        # self.optimizer_comboBox.setCurrentIndex(0)

        ## Create figure containers for the visualizations of data, training progress, and prediction
        self.figure_data = Figure()
        self.canvas_data = FigureCanvas(self.figure_data)
        self.toolbar_data = NavigationToolbar(self.canvas_data, self)
        layout_data = QVBoxLayout(self.data_graphicsView)
        layout_data.addWidget(self.canvas_data)
        layout_data.addWidget(self.toolbar_data)
        #
        self.figure_npv = Figure()
        self.canvas_npv = FigureCanvas(self.figure_npv)
        self.toolbar_npv = NavigationToolbar(self.canvas_npv, self)
        layout_npv = QVBoxLayout(self.npv_graphicsView)
        layout_npv.addWidget(self.canvas_npv)
        layout_npv.addWidget(self.toolbar_npv)
        #
        # self.figure = Figure()
        # self.canvas = FigureCanvas(self.figure)
        # self.toolbar = NavigationToolbar(self.canvas, self)
        # layout = QVBoxLayout(self.prediction_graphicsView)
        # layout.addWidget(self.canvas)
        # layout.addWidget(self.toolbar)
        #
        # # Initialize dropdown menus in Tab 1
        # self.utility_combo_box.addItems(["BGE", "PHI"])
        # self.utility_combo_box.currentIndexChanged.connect(self.update_company_combo_box)
        # self.utility_combo_box.currentIndexChanged.connect(self.update_id_combo_box)
        # self.utility_combo_box.setCurrentIndex(0)
        # utility_completer = QCompleter()
        # utility_completer.setModel(self.utility_combo_box.model())
        # utility_completer.setCaseSensitivity(Qt.CaseInsensitive)
        # self.utility_combo_box.setCompleter(utility_completer)
        # self.utility_combo_box.setEditable(True)
        #
        # self.building_combo_box.addItems(["Public Charging", "Multifamily", "Residential"])
        # self.building_combo_box.currentIndexChanged.connect(self.update_company_combo_box)
        # self.building_combo_box.currentIndexChanged.connect(self.update_id_combo_box)
        # self.building_combo_box.setCurrentIndex(2)
        # building_completer = QCompleter()
        # building_completer.setModel(self.building_combo_box.model())
        # building_completer.setCaseSensitivity(Qt.CaseInsensitive)
        # self.building_combo_box.setCompleter(building_completer)
        # self.building_combo_box.setEditable(True)
        #
        # self.update_company_combo_box()
        # self.company_combo_box.currentIndexChanged.connect(self.update_id_combo_box)
        # self.company_combo_box.setCurrentIndex(0)
        # company_completer = QCompleter()
        # company_completer.setModel(self.company_combo_box.model())
        # company_completer.setCaseSensitivity(Qt.CaseInsensitive)
        # self.company_combo_box.setCompleter(company_completer)
        # self.company_combo_box.setEditable(True)
        #
        # self.update_id_combo_box()
        # self.id_combo_box.setCurrentIndex(3)
        # id_completer = QCompleter()
        # id_completer.setModel(self.id_combo_box.model())
        # id_completer.setCaseSensitivity(Qt.CaseInsensitive)
        # self.id_combo_box.setCompleter(id_completer)
        # self.id_combo_box.setEditable(True)
        #
        # # Initialize dropdown menus in Tab 2 (Data preprocessing)
        # self.utility_combo_box2.addItems(["BGE", "PHI"])
        # self.utility_combo_box2.currentIndexChanged.connect(self.update_company_combo_box2)
        # self.utility_combo_box2.setCurrentIndex(0)
        # utility_completer2 = QCompleter()
        # utility_completer2.setModel(self.utility_combo_box2.model())
        # utility_completer2.setCaseSensitivity(Qt.CaseInsensitive)
        # self.utility_combo_box2.setCompleter(utility_completer2)
        # self.utility_combo_box2.setEditable(True)
        #
        # self.building_combo_box2.addItems(["Public Charging", "Multifamily", "Residential"])
        # self.building_combo_box2.currentIndexChanged.connect(self.update_company_combo_box2)
        # self.building_combo_box2.setCurrentIndex(2)
        # building_completer2 = QCompleter()
        # building_completer2.setModel(self.building_combo_box2.model())
        # building_completer2.setCaseSensitivity(Qt.CaseInsensitive)
        # self.building_combo_box2.setCompleter(building_completer2)
        # self.building_combo_box2.setEditable(True)
        #
        # self.update_company_combo_box2()
        # self.company_combo_box2.setCurrentIndex(0)
        # company_completer2 = QCompleter()
        # company_completer2.setModel(self.company_combo_box2.model())
        # company_completer2.setCaseSensitivity(Qt.CaseInsensitive)
        # self.company_combo_box2.setCompleter(company_completer2)
        # self.company_combo_box2.setEditable(True)
        #
        # # Initialize other parameters
        # self.losses = []
        # self.target_date = "01/01/2022"
        # self.df = None
        # self.worker = None
        # self.session_data_path = None

    def reset(self):
        """
        Reset the entire program
        :return: None
        """
        self.clear_data()
        self.clear_model()
        self.clear_prediction()

    def select_data_file(self):
        options = QFileDialog.Options()
        self.data_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;CSV Files (*.csv);;XLSX Files (*.xlsx)",
                                                  options=options)
        # Set the file path to the QLineEdit
        if self.data_path:
            self.path_data_lineEdit.setText(self.data_path)

    def load_data(self):
        """
        Read the historical 1-hour load data for the customer specified.
         A customer is located based on utility, building type, company (charging service provider), and customer ID
        :return:
        """
        self.df_data = pd.DataFrame()
        if self.data_path:
            if ".csv" in self.data_path:
                self.df_data = pd.read_csv(self.data_path)
            elif ".xlsx" in self.data_path or ".xls" in self.data_path:
                self.df_data = pd.read_excel(self.data_path)
            else:
                QMessageBox.information(None, "Failed", "Data loading failed. File type is not supported.")

            if not self.df_data.empty:
                QMessageBox.information(None, "Success", "Raw session data is loaded successfully.")
                self.data_columns = list(self.df_data.columns)
                self.df_data["Time"] = pd.to_datetime(self.df_data["Time"])

        self.figure_data.clear()    # Clear the previous plot if any and visualize the newly loaded data
        ax = self.figure_data.add_subplot(111)
        ax.plot(self.df_data["Time"], self.df_data["Power(kW)"], label='Power(kW)')
        ax.set_ylabel("kW")
        #ax.xaxis_date()
        #ax.xaxis.set_major_locator(mdates.MonthLocator())
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Optional: Rotate the date labels for better readability
        #ax.tick_params(axis='x', rotation=45)
        ax.legend()
        self.canvas_data.draw()

        # Initialize worker for training after data loading
        #self.init_model()

    def load_obj(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def load_results(self):
        """
        Read the historical 1-hour load data for the customer specified.
         A customer is located based on utility, building type, company (charging service provider), and customer ID
        :return:
        """
        self.res_path = "results/constant_peak-Xu-flat.pkl"
        if self.res_path:
            res = self.load_obj(self.res_path)

        Emaxs = res["Emaxs"].copy()
        Pmaxs = res["Pmaxs"].copy()
        NPVs = res["NPVs"].copy()
        self.figure_npv.clear()    # Clear the previous plot if any and visualize the newly loaded data
        ax = self.figure_npv.add_subplot(111)

        X, Y = np.meshgrid(Emaxs, Pmaxs)
        X = X.transpose()
        Y = Y.transpose()
        levels = np.linspace(NPVs.min(), NPVs.max(), 31)
        plot = ax.contourf(X, Y, NPVs, levels=levels)
        self.figure_npv.colorbar(plot)
        ax.set_title('Net Present Value ($)')
        ax.set_ylabel('Rating of Power Equipment (kW)', fontsize=13)
        ax.set_xlabel('Initial Battery Energy Capacity (kWh)', fontsize=13)

        #ax.xaxis_date()
        #ax.xaxis.set_major_locator(mdates.MonthLocator())
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Optional: Rotate the date labels for better readability
        #ax.tick_params(axis='x', rotation=45)
        #ax.legend()
        self.canvas_npv.draw()

        # Initialize worker for training after data loading
        #self.init_model()

    def update_date(self, starting_date, ending_date):
        """
        Set the minimum and maximum date points allowed for prediction
        :param starting_date:
        :param ending_date:
        :return: None
        """
        date_format = "MM/dd/yyyy"
        Starting_Date = QDate.fromString(starting_date, date_format)
        Ending_Date = QDate.fromString(ending_date, date_format)
        self.predict_dateEdit.setMinimumDate(Starting_Date)
        #self.predict_dateEdit.setMaximumDate(Ending_Date)
        msg = f"Dates between {starting_date} and {ending_date} correspond to mock predictions, providing both actual and predicted values.\n" \
              f"Dates after {ending_date} will display only predicted values.\n" \
              f"Dates before {starting_date} are unsupported due to insufficient input data."
        self.predict_info_button.setToolTip(msg)
        #msg = f"Note: For predictions beyond {ending_date} where there is ground-truth;"
        #self.date_Label.setText(msg)

    def clear_data(self):
        """
        Clear the selected customer and the associated data
        :return: None
        """
        self.df = None
        self.utility = None
        self.building = None
        self.company = None
        self.customer = None
        self.starting_date_lineEdit.setText("-")
        self.ending_date_lineEdit.setText("-")
        self.total_energy_lineEdit.setText("-")
        self.peak_power_lineEdit.setText("-")
        self.figure_data.clear()
        self.canvas_data.draw()

    def update_learning_rate(self):
        """Get the scrollbar value and set the learning rate for training"""
        self.learning_rate = self.learning_rate_ScrollBar.value()/1000
        self.learning_rate_lineEdit.setText(str(self.learning_rate))
    def update_training_epoch(self):
        """Get the scrollbar value and set the training epochs for training"""
        self.training_epoch = self.training_epoch_ScrollBar.value()
        self.training_epoch_lineEdit.setText(str(self.training_epoch))
    def update_batch_size(self):
        """Get the scrollbar value and set the batch size for training"""
        self.batch_size = self.batch_size_ScrollBar.value()
        self.batch_size_lineEdit.setText(str(self.batch_size))
    def update_early_stopping(self):
        """Get the scrollbar value and set the early stopping epochs for training
        (This setting currently does not have an effect. Further development is reserved for future)"""
        self.early_stopping = self.early_stopping_ScrollBar.value()
        self.early_stopping_lineEdit.setText(str(self.early_stopping))

    def start_training(self):
        """
        Start training of the neural net
        :return:
        """
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        params = {} # Read parameters from GUI
        params["learning_rate"] = float(self.learning_rate_lineEdit.text())
        params["batch_size"] = int(self.batch_size_lineEdit.text())
        params["training_epoch"] = int(self.training_epoch_lineEdit.text())
        params["early_stopping"] = int(self.early_stopping_lineEdit.text())
        params["optimizer"] = self.optimizer_comboBox.currentText()

        self.training_epoch = params["training_epoch"]
        self.losses = []

        # Start training (in a separate thread "worker")
        self.worker.param_loader(params)
        self.worker.progress_epoch.connect(self.update_epoch)
        self.worker.progress_loss.connect(self.update_loss)
        self.worker.training_stopped.connect(self.on_training_stopped)
        self.worker.training_done.connect(self.on_training_done)
        self.worker.start()

    def on_training_done(self):
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        QMessageBox.information(None, "Success", "Training is done!")

    def stop_training(self):
        self.worker.stop()

    def on_training_stopped(self):
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        QMessageBox.information(None, "Success", "Training was stopped.")

    def update_epoch(self, epoch):
        """
        Update training progress bar
        :return: None
        """
        pct = epoch / (self.training_epoch-1) * 100
        pct = int(pct)
        self.progress_bar.setValue(pct)  # MNIST dataset has 938 batches

    def update_loss(self, loss_epoch):
        """
        Update training loss plot
        :return: None
        """
        self.losses.append(loss_epoch)
        if len(self.losses) % 10 == 0:
            self.figure_train.clear()
            ax = self.figure_train.add_subplot(111)
            ax.plot(self.losses, label='Training Loss')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            self.canvas_train.draw()

    def init_model(self):
        """
        Initiate model and pass the parameters and data to the model
        :return: None
        """
        params = {}
        params["learning_rate"] = float(self.learning_rate_lineEdit.text())
        params["batch_size"] = int(self.batch_size_lineEdit.text())
        params["training_epoch"] = int(self.training_epoch_lineEdit.text())
        params["early_stopping"] = int(self.early_stopping_lineEdit.text())
        params["optimizer"] = self.optimizer_comboBox.currentText()

        self.worker = Worker()
        self.worker.param_loader(params)
        try:
            self.worker.data_loader(self.df)
        except:
            pass

    def clear_model(self):
        """
        If there is an existing model, it will be reset
        :return: None
        """
        self.init_model()
        self.figure_train.clear()
        self.canvas_train.draw()
        self.progress_bar.setValue(0)

    def save_model(self):
        """
        Save the trained model
        :return: None
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name_temp = f"{self.utility}-{self.building}-{self.company}-{self.customer}"
        file_name, _ = QFileDialog.getSaveFileName(None, "Save Model", file_name_temp, "Torch Model Files (*.pt);;All Files (*)",
                                                   options=options)
        # If a file name is selected, save the model
        if file_name:
            if not file_name.endswith('.pt'):
                file_name += '.pt'
            try:
                self.worker.save(file_name)
                QMessageBox.information(None, "Success", "Model saved successfully!")
            except:
                QMessageBox.information(None, "Failed", "Model saving failed.")

    def load_model(self):
        """
        Load an existing model
        :return: None
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(None, "Save Model", "", "Torch Model Files (*.pt);;All Files (*)",
                                                   options=options)
        # If a file name is selected, load the model
        if file_name:
            try:
                self.worker.load(file_name)
                QMessageBox.information(None, "Success", "Model loaded successfully!")
            except:
                QMessageBox.information(None, "Failed", "Model loading failed.")

    def predict(self):
        """
        Make prediction based on a trained model or a loaded model
        :return: None
        """
        if self.df is None:
            QMessageBox.information(None, "Failed", "There is no data loaded. Please load data before prediction.")
            return
        if self.worker.model is None:
            QMessageBox.information(None, "Failed", "There is no model. Please train or load a model before prediction.")
            return

        self.target_date = self.predict_dateEdit.date().toString('MM/dd/yyyy')
        self.worker.predict(self.target_date)
        x = pd.date_range(self.target_date, periods=24, freq="H")
        self.y_pred = np.clip(self.worker.y_pred, 0, None)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, self.y_pred, label='prediction')
        if self.worker.y_actual:
            self.y_actual = self.worker.y_actual.copy()
            ax.plot(x, self.y_actual, label='actual')
        ax.set_ylabel("kW")
        ax.set_xlabel("Hour of the day")
        mse = np.mean(np.square(self.y_pred - self.y_actual))
        mse = round(mse, 2)
        self.MSE_Label.setText(f"Mean Squared Error: {mse} kW")

        ax.legend()
        self.canvas.draw()
        self.export_button.setEnabled(True)

    def clear_prediction(self):
        self.y_pred = None
        self.y_actual = None
        self.figure.clear()
        self.canvas.draw()

    def export(self):
        """
        Export the prediction results, including Time, Actual usage (if applicable), and prediction
        :return:
        """
        # Show a QFileDialog to select the location to save the results
        date = self.target_date.replace("/", "-")
        file_name = f"{self.utility}-{self.building}-{self.company}-{self.customer}-{date}.csv"
        results_file, _ = QFileDialog.getSaveFileName(self, "Save Results", file_name, "CSV Files (*.csv)")

        if results_file:
            # # Display the selected directory
            # self.directory_line_data.setText(results_file)
            # Save the prediction series to the file
            x = pd.date_range(start=self.target_date, periods=24, freq='H')
            if self.y_actual:
                df = pd.DataFrame([x, self.y_actual, self.y_pred]).transpose()
                df.columns = ["Time", "Actual (kW)", "Prediction (kW)"]
            else:
                df = pd.DataFrame([x, self.y_pred]).transpose()
                df.columns = ["Time", "Prediction (kW)"]
            df.to_csv(results_file, index=False)
            QMessageBox.information(None, "Success", "Prediction results exported successfully.")

    def select_session_data_file(self):
        options = QFileDialog.Options()
        self.session_data_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;CSV Files (*.csv);;XLSX Files (*.xlsx)",
                                                  options=options)
        # Set the file path to the QLineEdit
        if self.session_data_path:
            self.path_session_data_lineEdit.setText(self.session_data_path)

    def load_session_data(self):
        self.df_session = pd.DataFrame()
        if self.session_data_path:
            if ".csv" in self.session_data_path:
                self.df_session = pd.read_csv(self.session_data_path)
            elif ".xlsx" in self.session_data_path or ".xls" in self.session_data_path:
                self.df_session = pd.read_excel(self.session_data_path)
            else:
                QMessageBox.information(None, "Failed", "Data loading failed. File type is not supported.")

            if not self.df_session.empty:
                QMessageBox.information(None, "Success", "Raw session data is loaded successfully.")
                self.session_data_columns = list(self.df_session.columns)

    def preprocessing(self):
        ts_min = pd.Timestamp('2024-01-01 00:00:00').tz_localize('US/Eastern')
        te_max = pd.Timestamp('2000-01-01 00:00:00').tz_localize('US/Eastern')
        # Use US Federal Holiday Calendar
        calendar = USFederalHolidayCalendar()
        holidays = calendar.holidays(start='2000-01-01', end='2030-12-31')
        holidays = [holiday.date() for holiday in holidays]

        self.utility2 = self.utility_combo_box2.currentText()
        self.building2 = self.building_combo_box2.currentText()
        self.company2 = self.company_combo_box2.currentText()

        id_col = self.id_col_comboBox.currentText()
        start_time_col = self.start_time_col_comboBox.currentText()
        end_time_col = self.end_time_col_comboBox.currentText()
        total_energy_col = self.total_energy_col_comboBox.currentText()
        cols = [id_col, start_time_col, end_time_col, total_energy_col]
        df1 = self.df_session[cols]
        df1 = df1.rename(columns={id_col: "ID", \
                                 start_time_col: "Start Time", \
                                 end_time_col: "End Time", \
                                 total_energy_col: "Energy(kWh)", \
                                 })
        IDs = list(df1["ID"].unique())
        number_of_IDs = len(IDs)
        for count, ID in enumerate(IDs):
            pct = (count + 1) / number_of_IDs * 100
            pct = int(pct)
            self.data_progress_bar.setValue(pct)

            ind = df1["ID"] == ID
            df = df1.loc[ind]
            df = df.drop_duplicates(subset=['Start Time'])
            df["Start Time"] = pd.to_datetime(df["Start Time"])
            df["Start Time"] = df["Start Time"].dt.tz_localize('US/Eastern', ambiguous=False,
                                                               nonexistent="shift_backward")
            df["End Time"] = pd.to_datetime(df["End Time"])
            df["End Time"] = df["End Time"].dt.tz_localize('US/Eastern', ambiguous=False, nonexistent="shift_forward")
            df = df.sort_values('Start Time')
            ind = df['Start Time'].notna() & df['End Time'].notna()
            df = df.loc[ind]
            durations_hour = (df["End Time"] - df["Start Time"]).dt.total_seconds() / 3600
            df["Power(kW)"] = df["Energy(kWh)"] / durations_hour

            t0 = df["Start Time"].iloc[0].floor("H")
            t1 = df["End Time"].iloc[-1].ceil("H")
            num_minutes = (t1 - t0).total_seconds() / 60
            num_minutes = int(num_minutes)

            tss = df["Start Time"] - t0
            tss = tss.round("min").dt.total_seconds() / 60
            tss = tss.astype(int)

            tes = df["End Time"] - t0
            tes = tes.round("min").dt.total_seconds() / 60
            tes = tes.astype(int)

            if t0 < ts_min:
                ts_min = t0
            if t1 > te_max:
                te_max = t1

            # Get 1-min average power
            powers = df["Power(kW)"]
            ps = np.zeros(num_minutes)

            for i in tss.index:
                ts = tss[i]
                te = tes[i]
                power = powers[i]
                ps[ts:te] = power

            # Get 1-hour average power
            ps1h = np.mean(ps.reshape(-1, 60), axis=1)
            time_points = [t0 + timedelta(seconds=3600 * i) for i in range(len(ps1h))]

            df = pd.DataFrame({"Time": time_points, "Power(kW)": ps1h})
            df["Time"] = df["Time"].astype(str).apply(lambda x: x[:-6])  # Get rid of time zone for now
            # Adding time zone by infer, otherwise, the resultant Time column will not be pd timestamp
            df["Time"] = pd.to_datetime(df["Time"]).dt.tz_localize('US/Eastern', ambiguous="infer")

            df["hour"] = df["Time"].dt.hour
            df["sin_hour"] = np.sin(df["hour"] / 24 * 2 * np.pi)
            df["cos_hour"] = np.cos(df["hour"] / 24 * 2 * np.pi)

            df["dayofweek"] = df["Time"].dt.dayofweek
            df["weekday"] = df["Time"].dt.dayofweek.apply(lambda x: 1 if x <= 4 else 0)
            df['holiday'] = df['Time'].dt.date.isin(holidays).apply(lambda x: 1 if x else 0)

            dst = os.path.join("data", self.utility2, self.building2, self.company2, f"{ID}.csv")
            if os.path.isfile(dst): # if file exist, append
                df0 = pd.read_csv(dst)
                df = pd.concat([df, df0])
            df.to_csv(dst, index=False)
        QMessageBox.information(None, "Success", f"Data Preprocessing finished.\nThere are {number_of_IDs} unique IDs processed.")

    def update_company_combo_box(self):
        # Get the currently selected utility and building
        utility = self.utility_combo_box.currentText()
        building = self.building_combo_box.currentText()

        # Determine the directory based on the selected utility and building
        directory = os.path.join("data", utility, building)

        # Clear the company_combo_box
        if building == "Public Charging":
            self.company_combo_box.setEnabled(False)
        else:
            self.company_combo_box.setEnabled(True)
            self.company_combo_box.clear()

            # List the contents of the directory
            if os.path.exists(directory):
                for dir_name in os.listdir(directory):
                    if os.path.isdir(os.path.join(directory, dir_name)):
                        # If it is a directory, add its name to the company_combo_box
                        self.company_combo_box.addItem(dir_name)

    def update_company_combo_box2(self):
        # Get the currently selected utility and building
        utility = self.utility_combo_box2.currentText()
        building = self.building_combo_box2.currentText()

        # Determine the directory based on the selected utility and building
        directory = os.path.join("data", utility, building)

        # Clear the company_combo_box
        if building == "Public Charging":
            self.company_combo_box2.setEnabled(False)
        else:
            self.company_combo_box2.setEnabled(True)
            self.company_combo_box2.clear()

            # List the contents of the directory
            if os.path.exists(directory):
                for dir_name in os.listdir(directory):
                    if os.path.isdir(os.path.join(directory, dir_name)):
                        # If it is a directory, add its name to the company_combo_box
                        self.company_combo_box2.addItem(dir_name)

    def update_id_combo_box(self):
        # Get the currently selected utility, building, and company
        utility = self.utility_combo_box.currentText()
        building = self.building_combo_box.currentText()
        company = self.company_combo_box.currentText()

        # Determine the directory based on the selected utility, building, and company
        directory = directory = os.path.join("data", utility, building, company)

        # Clear the id_combo_box
        self.id_combo_box.clear()

        # List the contents of the directory
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith(".csv"):
                    # If the file is a CSV file, add its name (without the extension) to the id_combo_box
                    id = os.path.splitext(file)[0]
                    self.id_combo_box.addItem(id)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ApplicationWindow()
    win.show()
    sys.exit(app.exec_())
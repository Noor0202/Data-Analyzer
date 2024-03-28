# Importing all necessary Libraries

from scipy.stats import skew,kurtosis
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import numpy as np
import subprocess
import math

# This class contains methods like freq table for both discrete and continuous.
# It takes data and gives mean, variance, and other stats about data which are used in distribution.
class DistributionAnalyzer:
    
    # Initialize stats of data
    def __init__(self,data):
        self.__data = data
        self.sum_data = sum(data)
        self.len_data = len(self.__data)
        self.mean = np.mean(data) 
        self.variance = np.var(data)
        self.standard_deviation = np.std(data)
        self.log_mean = sum(np.log(x) for x in self.__data) / self.len_data
        self.log_variance = sum((np.log(x) - self.log_mean) ** 2 for x in self.__data) / self.len_data
        self.log_std = np.sqrt(self.log_variance)
        self.coefficient_of_variation = np.sqrt(self.variance) / self.mean
        self.tau = self.standard_deviation / self.mean
        self.laxious_ratio = self.standard_deviation / self.mean
        self.skewness = skew(self.__data)
        self.kurtosis = kurtosis(self.__data)
        
    # Frequency table for discrete data which calculates observed frequency
    def discrete_freq_table(self):
        data_set = set(self.__data)
        intervals_list = []
        for i in data_set:
            interval = [i, 0, self.__data.count(i)]
            intervals_list.append(interval)
        return intervals_list
    
    # Frequency table for continuous data which calculates observed frequency
    def continuous_freq_table(self, no_of_interval):
        upper_bound = max(self.__data)
        lower_bound = min(self.__data)
        class_interval = (upper_bound - lower_bound) / no_of_interval
        intervals_list = []
        for i in range(no_of_interval):
            upper_bound = lower_bound + class_interval
            interval = [lower_bound, upper_bound]
            interval.append(sum(1 for value in self.__data if lower_bound - 1e-10 <= value < upper_bound + 1e-10))
            lower_bound = upper_bound
            intervals_list.append(interval)
        return intervals_list
    
    # It creates observed frequency and intervals based on function used in exponential and Weibull distributions
    def interval_boundaries(self,expo_function, ei_freq, no_of_intervals):
        intervals = []
        tolerance = 1e-10
        prev_interval_key = [0, 0]

        for i in range(1,no_of_intervals + 1):
            upper_bound = expo_function(i)
            if i == 1:
                interval_key = [0, upper_bound]
                prev_interval_key = interval_key
            else:
                interval_key = [prev_interval_key[1], upper_bound]

            of_freq = sum(1 for value in self.__data if interval_key[0] - tolerance <= value < interval_key[1] + tolerance)

            prev_interval_key = interval_key
            
            interval_key.append(of_freq)
            interval_key.append(ei_freq)
            interval_key.append(0)
            intervals.append(interval_key)

        return intervals

    # Calculating MLE for gamma distribution
    def estimate_maximum_likelihood(self):
        total_mean = 0
        for i in self.__data:
            total_mean += np.log(i)
        t_value = 1 / (np.log(self.mean) - (total_mean) / len(self.__data))
        return t_value

    # Area under the curve for all continuous distributions
    def area_under_the_curve(function, lower_bound, upper_bound, no_sub_intervals):
        width = (upper_bound - lower_bound) / no_sub_intervals
        total_area = 0

        for i in range(no_sub_intervals):
            x0 = lower_bound + i * width
            x1 = lower_bound + (i + 1) * width
            try:
                area = ((function(x0) + function(x1)) * width) / 2
                total_area += area
            except ZeroDivisionError:
                print("Warning: ZeroDivisionError occurred. Discrete data.")
            except NameError as e:
                print(f"Warning: NameError occurred - {e}")

        return total_area


    # Merged interval method for all distributions
    def merged_intervals(list_intervals, threshold):
        merg_intern = []
        curr_interval = None
        of_total = 0
        ef_total = 0
        area_total = 0

        for sub_list in list_intervals:
            lower_bound, upper_bound, obs_freq, exp_freq, area = sub_list
            if exp_freq < threshold:
                if curr_interval is None:
                    curr_interval = [lower_bound, upper_bound]
                    of_total = obs_freq
                    ef_total = exp_freq
                    area_total = area
                else:
                    curr_interval = [curr_interval[0], upper_bound]
                    of_total += obs_freq
                    ef_total += exp_freq
                    area_total += area
            else:
                if curr_interval is not None:
                    merg_intern.append([curr_interval[0], curr_interval[1], of_total, ef_total,area_total])
                    curr_interval = None
                    of_total = 0
                    ef_total = 0
                    area_total = 0
                    
                merg_intern.append([lower_bound, upper_bound, obs_freq, exp_freq, area])

        if curr_interval is not None:
            merg_intern.append([curr_interval[0], curr_interval[1], of_total, ef_total, area_total])

        return merg_intern
    
    # Calculating t_value for gamma distribution
    def find_t_value(t_value):
        
        t_value_data = {
            0.01: 0.010, 0.02: 0.019, 0.03: 0.027, 0.04: 0.036, 0.05: 0.044, 0.06: 0.052, 0.07: 0.060, 0.08: 0.068,0.09: 0.076, 0.10: 0.083, 0.11: 0.090, 0.12: 0.098, 0.13: 0.105, 0.14: 0.112, 0.15: 0.119, 0.16: 0.126,0.17: 0.133, 0.18: 0.140, 0.19: 0.147, 0.20: 0.153, 0.30: 0.218, 0.40: 0.279, 0.50: 0.338, 0.60: 0.396,0.70: 0.452, 0.80: 0.507, 0.90: 0.562, 1.00: 0.616, 1.10: 0.669, 1.20: 0.722, 1.30: 0.775, 1.40: 0.827,1.50: 0.879, 1.60: 0.931, 1.70: 0.983, 1.80: 1.035, 1.90: 1.086, 2.00: 1.138, 2.10: 1.189, 2.20: 1.240,2.30: 1.291, 2.40: 1.342, 2.50: 1.393, 2.60: 1.444, 2.70: 1.495, 2.80: 1.546, 2.90: 1.596, 3.00: 1.647,3.10: 1.698, 3.20: 1.748, 3.30: 1.799, 3.40: 1.849, 3.50: 1.900, 3.60: 1.950, 3.70: 2.001, 3.80: 2.051,3.90: 2.101, 4.00: 2.152, 4.20: 2.253, 4.40: 2.353, 4.60: 2.424, 4.80: 2.554, 5.00: 2.655, 5.20: 2.755,5.40: 2.856, 5.60: 2.956, 5.80: 3.057, 6.00: 3.157, 6.20: 3.257, 6.40: 3.357, 6.60: 3.458, 6.80: 3.558,7.00: 3.658, 7.20: 3.759, 7.40: 3.859, 7.60: 3.959, 7.80: 4.059, 8.00: 4.159, 8.20: 4.260, 8.40: 4.360,8.60: 4.460, 8.80: 4.560, 9.00: 4.660, 9.20: 4.760, 9.40: 4.860, 9.60: 4.961, 9.80: 5.061, 10.00: 5.161,10.50: 5.411, 11.00: 5.661, 11.50: 5.912, 12.00: 6.162, 12.50: 6.412, 13.00: 6.662, 13.50: 6.912, 14.00: 7.163,14.50: 7.413, 15.00: 7.663, 15.50: 7.913, 16.00: 8.163, 16.50: 8.413, 17.00: 8.663, 17.50: 8.913, 18.00: 9.163,18.50: 9.414, 19.00: 9.664, 19.50: 9.914, 20.00: 10.164, 20.50: 10.414, 21.00: 10.664, 21.50: 10.914,22.00: 11.164, 22.50: 11.414, 23.00: 11.664, 23.50: 11.914, 24.00: 12.164, 24.50: 12.414, 25.00: 12.664,30.00: 15.165, 35.00: 17.665, 40.00: 20.165, 45.00: 22.665, 50.00: 25.165 }

            
        closest_diff = float('inf') 
        closest_value = None

        if t_value < 0.01:
            return t_value_data[0.01]
            
        if t_value > 50.00:
            return t_value_data[50.00]
            
        for key in t_value_data.keys():
            diff = abs(key - t_value)
            if diff < closest_diff:
                closest_diff = diff
                closest_value = t_value_data[key]

        return closest_value

    # Checking which distribution is best for data based on above stats 
    def check_distribution(self):
        
        print("\n\n--NOTE--".center(80))
        print("\nI'm using statistical methods to select the best-fitting distribution for the data based on feet measurements, but I recommend visualizing the data and analyzing PDF graphs for accuracy.\n".center(80))

        
        dis_option = []
        
        if type(self.__data[0]) is float or type(self.__data[0]) is np.float64:
            print("\nData seems to be Continuous data.")
            print("\n\tYou have some distribution options:")
            print("\n\t\t1) Normal Distribution\n\t\t2) Exponential Distribution\n\t\t3) Gamma Distribution\n\t\t4) Log-Normal Distribution\n\t\t5) Weibull Distribution\n\t\t6) Beta Distribution")
            
            if math.ceil(self.coefficient_of_variation) == 1 or math.ceil(self.skewness) == 2 :
                dis_option.append(f"Exponential Distribution -\tCV ({self.coefficient_of_variation}) is close to 1 or Skewness ({self.skewness}) is near to 2")
                
            if self.skewness > 0:
                dis_option.append(f"Gamma Distribution -\tdistribution is skewed to the right -\t Skewness ({self.skewness})")
                dis_option.append(f"Log Normal Distribution -\tdistribution is skewed to the right -\t Skewness ({self.skewness})")
            
            if math.ceil(self.skewness) == 0:
                dis_option.append(f"Normal Distribution -\tdistribution is approximately symmetric -\t Skewness ({self.skewness})")
                
            
            if min(self.__data) >= 0 and max(self.__data) <= 1:
                dis_option.append(f"Beta Distribution -\t0 <= value <= 1 - Min - {min(self.__data)}\t Max - {max(self.__data)}")
                
        else:
            print("\nData seems to be Discrete data.")
            print("\n\tYou have some distribution options:")
            print("\n\t\t7) Binomial Distribution\n\t\t8) Poisson Distribution\n\t\t9) Geometric Distribution\n\t\t10) Negative Binomial Distribution")
            
            if math.ceil(self.tau) == 1:
                dis_option.append(f"Poisson Distribution -\tTau ({self.tau}) is close to 1")
                
            if self.tau > 1:
                dis_option.append(f"Negative Binomial Distribution-\tTau ({self.tau}) is > 1")
            
            if self.tau < 1:
                dis_option.append(f"Binomial Distribution-\tTau ({self.tau}) is < 1")
        
        print("\n\t Here are Some Suggestions based on your data :\n")
        for i in range(len(dis_option)):
            print("\t\t - ", dis_option[i])
        print("\n\n")
    
    # Plotting scatter before selecting distribution
    def plot_data_scatter(self):
        
        except_last = self.__data[:-1]
        except_first = self.__data[1:]
        
        scatter_intervals_list = self.discrete_freq_table()
        
        x_values = [sub_list[0] for sub_list in scatter_intervals_list]        
        y_values = [sub_list[2] for sub_list in scatter_intervals_list]        
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, color='cyan')
        plt.scatter(except_last, except_first, color='orange')
        plt.xlabel('Data Points')
        plt.ylabel('Observed Frequency')
        plt.title('Scatter Plot of Data')
        plt.grid(True)
        plt.show()


# Class for formatting and outputting distribution analysis results.
class OutputFormatter(DistributionAnalyzer):
    
    def __init__(self, intervals_list,data):
        super().__init__(data)
        self.intervals_list = intervals_list
        
        self.__install_package()
        self.total_chi = self.__create_chi_sqr_table()
        self.error = False      
        
    # Private method to install required package 'tabulate'.
    def __install_package(self):
        try:
            subprocess.check_call(["pip", "install", "tabulate"])
        except subprocess.CalledProcessError:
            self.error = True
            
    # Private method to create the chi-square table.
    def __create_chi_sqr_table(self):
        
        of_total = 0
        ef_total = 0
        area_total = 0
        of_min_ef_total = 0
        of_min_ef_sqr_total = 0
        of_min_ef_sqr_div_ef_total = 0
        
        for i in range(len(self.intervals_list)):
            
            self.intervals_list[i].insert(0,i+1)
            
            ofminef = self.intervals_list[i][3] - self.intervals_list[i][4]
            
            self.intervals_list[i].append(ofminef)
            self.intervals_list[i].append(ofminef ** 2)
            self.intervals_list[i].append((ofminef ** 2) / self.intervals_list[i][4])
            
            of_total += self.intervals_list[i][3]
            ef_total += self.intervals_list[i][4]
            area_total += self.intervals_list[i][5]
            of_min_ef_total += float(self.intervals_list[i][6])
            of_min_ef_sqr_total += float(self.intervals_list[i][7])
            of_min_ef_sqr_div_ef_total += float(self.intervals_list[i][8])
        
        
        self.intervals_list.append(["Total",None,None,of_total,ef_total,area_total,of_min_ef_total,of_min_ef_sqr_total,of_min_ef_sqr_div_ef_total])
        
        return of_min_ef_sqr_div_ef_total
        
        
    # Private method to print detailed interval table.
    def __print_interval_table_detail(self):
        
        df = pd.DataFrame(self.intervals_list, columns = ["  NO", "  LB", "  UB", "  OF", "  EF", "AREA  ", "(OF - EF)  ", "(OF - EF)^2  ", "(OF - EF)^2 / EF"])
        print(df)
    
    # Method to print the chi-square table.
    def print_chi_square_table(self):
        
        if self.error:
            self.__print_interval_table_detail()
        else:
            print(tabulate(self.intervals_list, headers=["NO", "LB", "UB", "OF", "EF", "AREA", "(OF - EF)", "(OF - EF)^2", "(OF - EF)^2 / EF"], tablefmt="grid"))
    
    # Method to print statistics of the data.
    def print_stats(self):
        print("\n\nTotal Length of Data - ", self.len_data)
        print("Mean of Data - ", self.mean)
        print("Logarithmic Mean of Data - ", self.log_mean)
        print("Variance of Data - ", self.variance)
        print("Logarithmic Variance of Data - ", self.log_variance)
        print("Standard Deviation of Data - ", self.standard_deviation)
        print("Logarithmic Standard Deviation of Data - ", self.log_std)
        print("Coefficient of Variation of Data - ", self.coefficient_of_variation)
        print("Tau Value of Data - ", self.tau)
        print("Skewness of Data - ", self.skewness)
        print("Kurtosis of Data - ", self.kurtosis)
        print("Laxious Ratio of Data - ",self.laxious_ratio,"\n\n")
        
    # Method to find the chi-square table value.
    def find_chi_sqr_table_value(self,degree_of_freedom,level_of_significance):
        chi_sqr = [[None, 0.99, 0.975, 0.95, 0.9, 0.1, 0.05, 0.025, 0.01],[1, None, 0.001, 0.004, 0.016, 2.706, 3.841, 5.024, 6.635],[2, 0.020, 0.051, 0.103, 0.211, 4.605, 5.991, 7.378, 9.21],[3, 0.115, 0.216, 0.352, 0.584, 6.251, 7.815, 9.348, 11.345],[4, 0.297, 0.484, 0.711, 1.064, 7.779, 9.488, 11.143, 13.277],[5, 0.554, 0.831, 1.145, 1.61, 9.236, 11.07, 12.833, 15.086],[6, 0.872, 1.237, 1.635, 2.204, 10.645, 12.592, 14.449, 16.812],[7, 1.239, 1.69, 2.167, 2.833, 12.017, 14.067, 16.013, 18.475],[8, 1.646, 2.18, 2.733, 3.49, 13.362, 15.507, 17.535, 20.09],[9, 2.088, 2.7, 3.325, 4.168, 14.684, 16.919, 19.023, 21.666],[10, 2.558, 3.247, 3.94, 4.865, 15.987, 18.307, 20.483, 23.209],[11, 3.053, 3.816, 4.575, 5.578, 17.275, 19.675, 22.92, 24.725],[12, 3.571, 4.404, 5.226, 6.304, 18.549, 21.026, 23.337, 26.217],[13, 4.107, 5.009, 5.892, 7.042, 19.812, 22.362, 24.736, 27.688],[14, 4.66, 5.629, 6.571, 7.79, 21.064, 23.685, 26.119, 29.141],[15, 5.229, 6.262, 7.261, 8.547, 22.307, 24.996, 27.488, 30.578],[16, 5.812, 6.908, 7.962, 9.312, 23.542, 26.296, 28.845, 32.0],[17, 6.408, 7.564, 8.672, 10.085, 24.769, 27.587, 30.191, 33.409],[18, 7.015, 8.231, 9.39, 10.865, 25.989, 28.869, 31.526, 34.805],[19, 7.633, 8.907, 10.117, 11.651, 27.204, 30.144, 32.852, 36.191],[20, 8.26, 9.591, 10.851, 12.443, 28.412, 31.41, 34.17, 37.566],[21, 8.897, 10.283, 11.591, 13.24, 29.615, 32.671, 35.479, 38.932],[22, 9.542, 10.982, 12.338, 14.041, 30.813, 33.924, 36.781, 40.289],[23, 10.196, 11.689, 13.091, 14.848, 32.007, 35.172, 38.076, 41.638],[24, 10.856, 12.401, 13.848, 15.659, 33.196, 36.415, 39.364, 42.98],[25, 11.524, 13.12, 14.611, 16.473, 34.382, 37.652, 40.646, 44.314],[26, 12.198, 13.844, 15.379, 17.292, 35.563, 38.885, 41.923, 45.642],[27, 12.879, 14.573, 16.151, 18.114, 36.741, 40.113, 43.195, 46.963],[28, 13.565, 15.308, 16.928, 18.939, 37.916, 41.337, 44.657, 48.278],[29, 14.256, 16.047, 17.708, 19.768, 39.087, 42.557, 45.722, 49.588],[30, 14.953, 16.791, 18.493, 20.599, 40.256, 43.773, 46.979, 50.892],[40, 22.164, 24.433, 26.509, 29.051, 51.805, 55.758, 59.342, 63.691],[50, 29.707, 32.357, 34.764, 37.689, 63.167, 67.505, 71.42, 76.154],[60, 37.485, 40.482, 43.188, 46.459, 74.397, 79.082, 83.298, 88.379],[70, 45.442, 48.758, 51.739, 55.329, 85.527, 90.531, 95.023, 100.425],[80, 53.54, 57.153, 60.391, 64.278, 96.578, 101.879, 106.629, 112.329],[100, 61.754, 65.647, 69.126, 73.291, 107.565, 113.145, 118.136, 124.116],[1000, 70.065, 74.222, 77.929, 82.358, 118.498, 124.342, 129.561, 135.807]]
        
        for i in range(len(chi_sqr)):
            if chi_sqr[i][0] == degree_of_freedom:
                for j in range(len(chi_sqr[0])):
                    if chi_sqr[0][j] == level_of_significance:
                        return chi_sqr[i][j]
        return 0
    
    # Method to check chi-square value against the chi-square table value.
    def check_chi_sqr_value(self, chi_sqr_table_value):
        
        print("\nTotal Chi Square Value - ", self.total_chi)
        print("\nChi Square Table Value - ", chi_sqr_table_value)
        
        if self.total_chi < chi_sqr_table_value:
            print(f"\nWe accept the hypothesis: data follows distribution (χ²: {self.total_chi} < {chi_sqr_table_value}).\n\nACCEPT DATA")
        else:
            print(f"\nWe reject the hypothesis: data does not follow distribution (χ²: {self.total_chi} > {chi_sqr_table_value}).\n\nREJECT DATA")
            
    # Method to plot continuous bar graph.
    def plot_continuous_bar(self):
        plt.figure(figsize=(10, 6))
        
        x_intervals = [f"{round(item[1],2)} - {round(item[2],2)}" for item in self.intervals_list if item[1] is not None and item[2] is not None]
        observed_freq = [item[3] for item in self.intervals_list[:-1]]
                
        plt.bar(x_intervals, observed_freq, width=0.5, label='Observed Frequency')
        plt.xlabel('Data Points')
        plt.ylabel('Observed Frequency')
        plt.xticks(rotation=45)
        plt.title('Bar Graph of Observed Frequency')
        plt.legend()
        plt.show()
    
    # Method to plot distribution bar graph.
    def plot_distribution_bar(self):
        
        plt.figure(figsize=(10, 6))
        
        self.intervals_list.sort(key = lambda x: x[1])
        
        x_intervals = [sub_list[1] for sub_list in self.intervals_list if sub_list[1] is not None]        
        observed_freq = [sub_list[3] for sub_list in self.intervals_list[:-1]]        
        
        plt.bar(x_intervals, observed_freq, width=0.5, label='Observed Frequency')
        plt.xlabel('Data Points')
        plt.ylabel('Observed Frequency')
        plt.xticks(rotation=45)
        plt.title('Bar Graph of Observed Frequency')
        plt.legend()
        plt.show()
        
    # Method to plot distribution bar graph of observed freq and expected freq.   
    def plot_observed_vs_expected_frequency(self):
    
        x_intervals = [item[0] for item in self.intervals_list[:-1]]
        observed_freq = [item[3] for item in self.intervals_list[:-1]]
        expected_freq = [item[4] for item in self.intervals_list[:-1]]
        plt.figure(figsize=(10, 6))
        plt.bar(x_intervals, observed_freq, width=0.3, label='Observed Frequency')
        plt.bar([x + 0.4 for x in x_intervals], expected_freq, width=0.3, label='Expected Frequency')
        plt.xlabel('Index')
        plt.ylabel('Frequency')
        plt.title('Bar Graph of Observed vs Expected Frequency')
        plt.legend()
        plt.show()

# Class for various probability distribution calculations.
class Distribution(DistributionAnalyzer):
    
    # Method to calculate gamma distribution.
    def gamma_distribution(self,number_of_interval):  
        
        intervals = DistributionAnalyzer.continuous_freq_table(self,number_of_interval)    
        beta = self.variance / self.mean  
        t_value = DistributionAnalyzer.estimate_maximum_likelihood(self)        
        alpha = DistributionAnalyzer.find_t_value(t_value)
        # alpha = self.mean / beta  
        
        def gamma_func(x):
            return (x ** (alpha - 1) * math.exp(-x))

        # gamma_value = DistributionAnalyzer.area_under_the_curve(gamma_func,0,65535,100000000)
        # error = 0
        
        gamma_value, error = quad(gamma_func, 0, float('inf'))
        
        def gamma_pdf(x):
            return (1 / (beta ** alpha) * gamma_value) * x ** (alpha - 1) * math.exp(-x / beta)
        

        for sub_list in intervals:
            total_area = DistributionAnalyzer.area_under_the_curve(gamma_pdf, sub_list[0], sub_list[1], 1000)
            expected_frequency = total_area * self.len_data
            sub_list.append(expected_frequency)
            sub_list.append(total_area)
        
        mer_inter = DistributionAnalyzer.merged_intervals(intervals,5)
        
        return (mer_inter,{"\n\nAlpha is - ":alpha,"Beta is - ":beta,"T_Value is - ":t_value,"Gamma_Value is - ":gamma_value,"Error is - ":error,"Degree of Freedom is - ":len(mer_inter) - 2 - 1})
    
    # Method to calculate normal distribution.
    def normal_distribution(self,number_of_interval):
        intervals = DistributionAnalyzer.continuous_freq_table(self,number_of_interval) 
                
        def norm_pdf(x):
            return (1 / math.sqrt(2 * math.pi * self.variance)) * (math.exp(-(x - self.mean) ** 2 / (2 * self.variance))) 
        
        for sub_list in intervals:
            total_area = DistributionAnalyzer.area_under_the_curve(norm_pdf, sub_list[0], sub_list[1], 1000000)
            expected_frequency = total_area * self.len_data
            sub_list.append(expected_frequency)
            sub_list.append(total_area)
        
        mer_inter = DistributionAnalyzer.merged_intervals(intervals,5)
        
        return (mer_inter,{"\n\nDegree of Freedom is - ":len(mer_inter) - 2 - 1})
    
    # Method to calculate log-normal distribution.
    def log_normal_distribution(self,number_of_interval):
        
        intervals = DistributionAnalyzer.continuous_freq_table(self,number_of_interval)
        
        def log_norm_pdf(x):
            return (1 / (x * self.log_std * math.sqrt(2 * math.pi))) * math.exp(-((np.log(x) - self.log_mean) ** 2) / (2 * self.log_std ** 2))
            
        for sub_list in intervals:
            total_area = DistributionAnalyzer.area_under_the_curve(log_norm_pdf, sub_list[0], sub_list[1], 1000000)
            expected_frequency = total_area * self.len_data
            sub_list.append(expected_frequency)
            sub_list.append(total_area)
        
        
        mer_inter = DistributionAnalyzer.merged_intervals(intervals,5)
        
        return (mer_inter,{"\n\nDegree of Freedom is - ":len(mer_inter) - 2 - 1})

    # Method to calculate beta distribution.
    def beta_distribution(self,number_of_interval):
        
        intervals = DistributionAnalyzer.continuous_freq_table(self,number_of_interval)
        
        alpha1 = self.mean * (((self.mean * (1 - self.mean)) / self.variance) - 1)
        alpha2 = (1 - self.mean) * (((self.mean * (1 - self.mean)) / self.variance) - 1)

        def beta_func(x):
            return  x ** (alpha1 - 1) * (1 - x) ** (alpha2 - 1)

        BETA_v, error = quad(beta_func, 0, 1)
        
        def beta_pdf(x):
            return ((x ** (alpha1 - 1) * (1 - x) ** (alpha2 - 1)) / BETA_v)

        for sub_list in intervals:
            total_area = DistributionAnalyzer.area_under_the_curve(beta_pdf, sub_list[0], sub_list[1], 1000000)
            expected_frequency = total_area * self.len_data
            sub_list.append(expected_frequency)
            sub_list.append(total_area)
        
        mer_inter = DistributionAnalyzer.merged_intervals(intervals,5)
        
        return (mer_inter,{"\n\nAlpha1 is - ":alpha1,"Alpha2 is - ":alpha2,"BETA_v is - ":BETA_v,"Error is - ":error,"Degree of Freedom is - ":len(mer_inter) - 2 - 1})
  
    # Method to calculate exponential distribution.
    def exponential_distribution(self,number_of_interval):
        
        lambdaa = 1 / self.mean
        expected_freq = self.len_data / number_of_interval
        p_value = expected_freq / self.len_data
        
        def expo_func(x):
            return (-1 / lambdaa) * (np.log(1 - x * p_value))
        
        intervals = DistributionAnalyzer.interval_boundaries(self,expo_func,expected_freq,number_of_interval)
        # print("\n\n",intervals) #error
        mer_inter = DistributionAnalyzer.merged_intervals(intervals,5)
        # print("\n\n",mer_inter)
        return (mer_inter,{"\n\nLambdaa is - ":lambdaa,"Expected Frequency is - ":expected_freq,"P_Value is - ":p_value,"Degree of Freedom is - ": len(mer_inter) - 1 - 1})  
    
    
    #pending
    def weibul_distribution(self,number_of_interval):
        pass      
    
    # Method to calculate binomial distribution.
    def binomial_distribution(self):

        intervals = DistributionAnalyzer.discrete_freq_table(self)

        prob_fail = self.variance / self.mean
        prob_succ = 1 - prob_fail
        no_of_trial = self.mean / prob_succ
        
        def bino_pdf(x):
            return math.comb(int(no_of_trial), x) * (prob_succ ** x) * (prob_fail ** (no_of_trial - x))       
    
        for sub_list in intervals:
            expected_frequency = bino_pdf(sub_list[0]) * self.len_data
            sub_list.append(expected_frequency)
            sub_list.append(0)
        
        mer_inter = DistributionAnalyzer.merged_intervals(intervals,5)
        
        return (mer_inter,{"\n\nProbability of Success is - ":prob_succ,"Probability of Failure is - ":prob_fail,"Number of Trial is - ":no_of_trial,"Degree of Freedom is - ": len(mer_inter) - 2 - 1})
    
    # Method to calculate Negative binomial distribution.
    def negative_binomial_distribution(self):
    
        intervals = DistributionAnalyzer.discrete_freq_table(self)
    
        prob_succ = self.mean / self.variance
        prob_fail = 1 - prob_succ
        no_of_success = math.ceil(self.mean * (prob_succ / prob_fail))
        
        def neg_bino_pdf(x):
            return math.comb(int(x + no_of_success - 1), int(x)) * ((1 - prob_succ) ** x) * (prob_succ ** no_of_success)
        
            
        for sub_list in intervals:
            expected_frequency = neg_bino_pdf(sub_list[0]) * self.len_data
            sub_list.append(expected_frequency)
            sub_list.append(0)
        
        mer_inter = DistributionAnalyzer.merged_intervals(intervals,5)
        
        return (mer_inter,{"\n\nProbability of Success is - ":prob_succ,"Probability of Failure is - ":prob_fail,"Number of Success is - ":no_of_success,"Degree of Freedom is - ": len(mer_inter) - 2 - 1})
    
    # Method to calculate Poisson distribution.
    def poisson_distribution(self):
        intervals = DistributionAnalyzer.discrete_freq_table(self)
        
        p_value = self.mean / self.len_data 
        
        def poison_pdf(x):
            return (math.exp(-self.mean) * self.mean ** x ) / math.factorial(x)        
        
        for sub_list in intervals:
            expected_frequency = poison_pdf(sub_list[0]) * self.len_data
            sub_list.append(expected_frequency)
            sub_list.append(0)
        
        mer_inter = DistributionAnalyzer.merged_intervals(intervals,5)
        
        return (mer_inter,{"\n\nProbability of Success is - ":p_value,"Degree of Freedom is - ": len(mer_inter) - 1 - 1})
    
    # Method to calculate Geometric distribution.
    def geometric_distribution(self):
        
        intervals = DistributionAnalyzer.discrete_freq_table(self)
        
        prob_succ = 1 / self.mean
                
        def geo_pdf(x):
            return (1 - prob_succ) ** (x - 1) * prob_succ

        for sub_list in intervals:
            expected_frequency = geo_pdf(sub_list[0]) * self.len_data
            sub_list.append(expected_frequency)
            sub_list.append(0)
        
        mer_inter = DistributionAnalyzer.merged_intervals(intervals,5)
        
        return (mer_inter,{"\n\nProbability of Success is - ":prob_succ,"Degree of Freedom is - ": len(mer_inter) - 1 - 1})    

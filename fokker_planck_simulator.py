import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cmath
import sympy as sp

class FokkerPlanckSimulator:
    def __init__(self, representation, simulation_time_setting, simulation_grid_setting, phys_parameter, init_cond, output_dir, ProbDensMap, solver, time_deriv_funcs, analytical=None):
        # Simulation time settings
        self.representation = representation
        self.t_start, self.t_end, self.dt = simulation_time_setting
        self.nsteps = int((self.t_end - self.t_start) / self.dt)
        self.t_vals = np.linspace(self.t_start, self.t_end, self.nsteps)

        # Grid for 2D probability representation value
        self.x, self.y = simulation_grid_setting

        # Initial conditions and solution storage
        self.phys_parameter = phys_parameter
        self.init_cond = init_cond

        self.solution = np.zeros((self.nsteps, len(init_cond)))
        self.solution[0] = init_cond

        # Functions for simulation
        self.ProbDensMap = ProbDensMap
        self.solver = solver
        self.time_deriv_funcs = time_deriv_funcs
        self.analytical = analytical

        # Prepare output directory
        self.output_dir = str('bin/')+output_dir

        # Initialize list to store centroid coordinates
        self.time = []
        self.centroid_x = []
        self.centroid_y = []

    def run_simulation(self, pure_parameter = False, compare_analytical = False):

        os.makedirs(self.output_dir, exist_ok=True)

        # Open a log file to write simulation messages
        log_filename = os.path.join(self.output_dir, "simulation_log.txt")
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Simulation started at {self.t_start} with time steps from {self.t_start} to {self.t_end}.\n")

            if pure_parameter:
                log_file.write("Running with pure_parameter mode enabled.\n")
                for t in tqdm(range(1, self.nsteps), desc="Simulating", unit="step"):
                    self.solution[t] = self.solver(self.t_vals[t-1], self.solution[t-1], self.dt, self.phys_parameter, self.time_deriv_funcs)
                # Plot the evolution of a(t), b(t), c(t), d(t) over time
                self.plot_parameter_evolution()
                log_file.write("Simulation completed.\n")
            else:
                # Precompute min and max values for consistent color scaling in plots
                os.makedirs(self.output_dir+str('/Snapshots'), exist_ok=True)
                if compare_analytical:
                    os.makedirs(self.output_dir+str('/AnalyticalSolution_Snapshots'), exist_ok=True)
                self.u_init = self.ProbDensMap(self.x, self.y, self.init_cond)
                self.vmin, self.vmax = np.min(self.u_init), np.max(self.u_init)

                log_file.write(f"Initial representation value range: vmin={self.vmin}, vmax={self.vmax}.\n")
                
                # Time integration using RK4 with progress bar
                for t in tqdm(range(1, self.nsteps), desc="Simulating", unit="step"):
                    self.solution[t] = self.solver(self.t_vals[t-1], self.solution[t-1], self.dt, self.phys_parameter, self.time_deriv_funcs)

                    # Compute the Representation value at the current time step
                    ProbDens = self.ProbDensMap(self.x, self.y, self.solution[t])
                    if compare_analytical:
                        AnaDens = self.analytical(self.x, self.y, (t)*self.dt, self.phys_parameter)
                        err = self.rms_error(AnaDens, ProbDens)
                        log_file.write(f"Comparison with analytical solution={err}.\n")

                    # Calculate the center of the probability distribution cloud
                    weighted_sum_x = np.sum(self.x[:, None] * ProbDens)  # Sum over x for each y
                    weighted_sum_y = np.sum(self.y[None, :] * ProbDens)  # Sum over y for each x
                    total_weight = np.sum(ProbDens)  # Total sum (normalization factor)

                    # Centroid coordinates
                    center_x = weighted_sum_x / total_weight
                    center_y = weighted_sum_y / total_weight

                    # Store the centroid coordinates
                    self.time.append(t * self.dt)
                    self.centroid_x.append(center_x)
                    self.centroid_y.append(center_y)

                    # Inside your run_simulation method (where the centroid and ProbDensSum are calculated)
                    ProbDensSum = self.volume_integration(self.x, self.y, ProbDens)

                    # Log the information and check for deviation from 1.0
                    self.log_warning_message(t, ProbDensSum, center_x, center_y, log_file)

                    # Save a snapshot every 10 steps
                    if t % 10 == 0:
                        self.save_snapshot(t, ProbDens,f'{self.output_dir}/Snapshots/snapshot_{t:04d}.png')
                        log_file.write(f"Snapshot saved for time step {t}.\n")
                        if compare_analytical:
                            self.save_snapshot(t, AnaDens,f'{self.output_dir}/AnalyticalSolution_Snapshots/snapshot_{t:04d}.png')

                # Plot the path of the center of the distribution at the end of the simulation
                self.plot_center_path()
                log_file.write("Path of the center plot generated.\n")

                # Plot the evolution of a(t), b(t), c(t), d(t) over time
                self.plot_parameter_evolution()
                log_file.write("Parameter evolution plots generated.\n")
            
            log_file.write(f"Simulation ended at {self.t_end}.\n")
            log_file.write("Simulation completed successfully.\n")
        
    
    def expectation_value(self, Exp_Func, t, realornot):
        X, Y = np.meshgrid(self.x, self.y)
        Alpha = X + 1j * Y
        Alpha_star = X - 1j * Y  
        t_idx = int(t/self.dt)
        if t_idx<0:
            t_idx = 0
        elif t_idx>=len(self.solution):
            t_idx = len(self.solution) - 1
        
        ProbDens = self.ProbDensMap(self.x, self.y, self.solution[t_idx]) 

        if realornot:
            return self.volume_integration(self.x, self.y, ProbDens * Exp_Func(Alpha, Alpha_star)).real
        else:
            return self.volume_integration(self.x, self.y, ProbDens * Exp_Func(Alpha, Alpha_star))
        
    def electric_field_evolution(self):
        def EF(alpha,alpha_star):
            return (alpha + alpha_star)/2.
        if self.representation == 'P':
            def EF2(alpha,alpha_star):
                return (alpha**2 + alpha_star**2 + 2 * alpha * alpha_star + 1)/4.
        elif self.representation == 'Q':
            def EF2(alpha,alpha_star):
                return (alpha**2 + alpha_star**2 + 2 * alpha * alpha_star - 1)/4.
        
        t = []
        EFs = []
        deltaEFs = []
        for i in range(len(self.solution)):
            t.append(i * self.dt)

            ef = self.expectation_value(EF,t[i],realornot=True)
            ef2 = self.expectation_value(EF2,t[i],realornot=True)

            EFs.append(ef)
            deltaEFs.append((ef2 - ef**2)**0.5)

        # 創建圖形
        plt.figure(figsize=(8, 6))

        # 繪製期望值 (EFs)
        plt.plot(t, EFs, label='Electric Field', color='b', marker='o', linestyle='-', markersize=6)

        # 繪製期望值的誤差條 (標準差)
        plt.errorbar(t, EFs, yerr=deltaEFs, fmt='o', label='Electric Field Standard Deviation', 
                    color='r', capsize=0.1, elinewidth=0.2, linestyle='None', markersize=0.1)

        # 設定圖標標題與軸標籤
        plt.title('Electric Field vs Time', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Electric Field Expectation Value', fontsize=12)

        # 顯示圖例
        plt.legend()

        # 顯示圖形
        plt.savefig(f'{self.output_dir}/electric_field_time_evolution.png')
        
    def volume_integration(self, x, y, Dens):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        Sum = np.sum(Dens) * np.abs(dx) * np.abs(dy)
        return Sum
    
    def rms_error(self, X,Y):
        return (np.sum((X - Y)**2)/np.sum(X)**2)**0.5
    
    def log_warning_message(self, t, ProbDensSum, center_x, center_y, log_file):
        # Calculate the difference from 1.0
        diff_from_one = abs(ProbDensSum - 1.0)

        # Check if the difference is more than 1% away from 1.0
        if diff_from_one > 0.01:
            log_file.write(f"WARNING: Step {t}, Time = {self.t_vals[t]:.2f}, ProbDensSum = {ProbDensSum:.4f}, Center = ({center_x:.4f}, {center_y:.4f})\nProbability represenation value summation on the plane has deviation larger than 1.0%! Simulation Box may be too small.\n")
        else:
            log_file.write(f"Step {t}, Time = {self.t_vals[t]:.2f}, ProbDensSum = {ProbDensSum:.4f}, Center = ({center_x:.4f}, {center_y:.4f})\n")
    
    def save_snapshot(self, t, ProbDens, name):
        # Create a plot for the Represenation Value
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(ProbDens.T, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower', aspect='auto', cmap='hot')#, vmin=self.vmin, vmax=self.vmax)
        ax.set_title(self.representation + f"-representation Time-evolution at Time = {t * self.dt:.2f}")
        ax.set_xlabel(r'Re{$\alpha$} (x)')
        ax.set_ylabel(r'Im{$\alpha$} (y)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Representation Value')

        # Plot the path of the center
        ax.plot(self.centroid_x, self.centroid_y, 'w-', label='Center Path', linewidth=2)
        ax.legend(loc="upper right")

        # Save snapshot
        plt.savefig(name)
        plt.close(fig)

    def plot_center_path(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.centroid_x, self.centroid_y, 'k-', linewidth=2)
        plt.xlabel(r'Re{$\alpha$}')
        plt.ylabel(r'Im{$\alpha$}')
        plt.title('Path of the Center of the Distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/Center_Path.png')
        plt.close()

        if len(self.time) == len(self.centroid_x):
            data = np.column_stack((self.time, self.centroid_x))
            np.savetxt(f'{self.output_dir}/time_centroid_x_values.txt', data, fmt=['%.6f', '%.6f'], delimiter='\t')
            print(f"Time and centroid_x values saved to {self.output_dir}/time_centroid_x_values.txt")

    def plot_parameter_evolution(self):
        plt.figure(figsize=(10, 8))
        n_params = len(self.solution[0])
        for i in range(min(n_params, 6)):
            plt.subplot(2, 3, i+1)
            plt.plot(self.t_vals, self.solution[:, i], label=f'param {i+1}(t)')
            plt.xlabel("Time")
            plt.ylabel(f'param {i+1}(t)')
            plt.title(f"Evolution of param {i+1}(t)")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/Parameter_evolution.png')
        plt.close()
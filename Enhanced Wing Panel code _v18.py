# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:13:35 2025

@author: hknar
"""
"""
Enhanced Wing Panel Method Analysis - Version 6
Fixed version addressing zero lift coefficient issue

Features:
- Airfoil coordinate input (NACA or custom)
- User-configurable span sections and chord panels
- Fixed boundary condition implementation 
- Corrected normal vector calculations
- Proper influence coefficient matrix
- Accurate angle of attack handling
- Educational parameter studies
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

class AirfoilGenerator:
    """Generate airfoil coordinates - NACA 4-digit series or custom input"""
    
    @staticmethod
    def naca_4digit(code, n_points=50):
        """Generate NACA 4-digit airfoil coordinates"""
        
        code_str = f"{code:04d}"
        m = int(code_str[0]) / 100.0    # Maximum camber
        p = int(code_str[1]) / 10.0     # Location of maximum camber
        t = int(code_str[2:4]) / 100.0  # Thickness
        
        # Generate x coordinates with cosine clustering
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))
        
        # Thickness distribution
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                      0.2843 * x**3 - 0.1015 * x**4)
        
        # Camber line
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        if m > 0 and p > 0:
            # Forward of maximum camber
            idx1 = x <= p
            yc[idx1] = m * x[idx1] / p**2 * (2 * p - x[idx1])
            dyc_dx[idx1] = 2 * m / p**2 * (p - x[idx1])
            
            # Aft of maximum camber
            idx2 = x > p
            yc[idx2] = m * (1 - x[idx2]) / (1 - p)**2 * (1 + x[idx2] - 2 * p)
            dyc_dx[idx2] = 2 * m / (1 - p)**2 * (p - x[idx2])
        
        # Surface coordinates
        theta = np.arctan(dyc_dx)
        
        # Upper surface
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        
        # Lower surface
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        
        return xu, yu, xl, yl
    
    @staticmethod
    def load_airfoil_file(filename):
        """Load airfoil coordinates from file"""
        try:
            data = np.loadtxt(filename, skiprows=1)  # Skip header
            x_coords = data[:, 0]
            y_coords = data[:, 1]
            
            # Separate upper and lower surfaces
            # Find leading edge (minimum x)
            le_idx = np.argmin(x_coords)
            
            # Upper surface (from LE to TE)
            xu = x_coords[le_idx::-1]  # Reverse to go LE to TE
            yu = y_coords[le_idx::-1]
            
            # Lower surface (from LE to TE)
            xl = x_coords[le_idx:]
            yl = y_coords[le_idx:]
            
            return xu, yu, xl, yl
            
        except Exception as e:
            print(f"Error loading airfoil file: {e}")
            print("Using NACA 0012 as default")
            return AirfoilGenerator.naca_4digit(12)

def get_user_inputs():
    """Get user inputs for wing analysis"""
    
    print("\n" + "="*60)
    print("ENHANCED WING PANEL METHOD ANALYSIS")
    print("="*60)
    print("Please provide the following parameters:")
    print("-"*40)
    
    # Wing geometry inputs
    print("\n1. WING GEOMETRY:")
    span = float(input("   Wing span (m) [default: 10.606]: ") or "10.606")
    root_chord = float(input("   Root chord (m) [default: 8.02]: ") or "8.02")
    tip_chord_ratio = float(input("   Tip chord ratio (tip/root) [default: 0.6]: ") or "0.6")
    tip_chord = root_chord * tip_chord_ratio
    sweep_angle = float(input("   Sweep angle (degrees) [default: 0]: ") or "0")
    dihedral_angle = float(input("   Dihedral angle (degrees) [default: 0]: ") or "0")
    
    # Airfoil selection
    print("\n2. AIRFOIL SELECTION:")
    print("   a) NACA 4-digit airfoil")
    print("   b) Load from file")
    print("   c) Flat plate (for validation)")
    
    airfoil_choice = input("   Select airfoil type [a/b/c, default: a]: ").lower() or "a"
    
    airfoil_data = None
    if airfoil_choice == 'a':
        naca_code = int(input("   Enter NACA 4-digit code [default: 2412]: ") or "2412")
        xu, yu, xl, yl = AirfoilGenerator.naca_4digit(naca_code)
        airfoil_data = {'xu': xu, 'yu': yu, 'xl': xl, 'yl': yl, 'type': f'NACA {naca_code}'}
        print(f"   Selected: NACA {naca_code}")
    
    elif airfoil_choice == 'b':
        filename = input("   Enter airfoil filename (with path): ")
        xu, yu, xl, yl = AirfoilGenerator.load_airfoil_file(filename)
        airfoil_data = {'xu': xu, 'yu': yu, 'xl': xl, 'yl': yl, 'type': f'File: {filename}'}
        print(f"   Loaded airfoil from: {filename}")
    
    else:
        # Flat plate
        x_flat = np.linspace(0, 1, 50)
        y_flat = np.zeros_like(x_flat)
        airfoil_data = {'xu': x_flat, 'yu': y_flat, 'xl': x_flat, 'yl': y_flat, 'type': 'Flat Plate'}
        print("   Selected: Flat Plate")
    
    # Panel discretization
    print("\n3. PANEL DISCRETIZATION:")
    n_span = int(input("   Number of spanwise panels per semi-span [default: 10]: ") or "10")
    n_chord = int(input("   Number of chordwise panels [default: 12]: ") or "12")
    
    print(f"   Total panels: {2 * n_span * n_chord} ({n_span} × {n_chord} per semi-span)")
    
    # Flight conditions
    print("\n4. FLIGHT CONDITIONS:")
    alpha = float(input("   Angle of attack (degrees) [default: 4]: ") or "4")
    V_inf = float(input("   Freestream velocity (m/s) [default: 1.0]: ") or "1.0")
    rho = float(input("   Air density (kg/m³) [default: 1.225]: ") or "1.225")
    
    # Analysis options
    print("\n5. ANALYSIS OPTIONS:")
    run_parameter_study = input("   Run parameter studies? [y/n, default: y]: ").lower() != 'n'
    create_plots = input("   Create visualizations? [y/n, default: y]: ").lower() != 'n'
    debug_mode = input("   Enable debug output? [y/n, default: n]: ").lower() == 'y'
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY:")
    print(f"Wing: {span}m span, {root_chord}m root chord, taper ratio {tip_chord_ratio}")
    print(f"Airfoil: {airfoil_data['type']}")
    print(f"Panels: {2 * n_span * n_chord} total ({n_span} spanwise × {n_chord} chordwise per semi-span)")
    print(f"Conditions: α = {alpha}°, V∞ = {V_inf} m/s")
    print("="*60)
    
    return {
        'span': span,
        'root_chord': root_chord,
        'tip_chord': tip_chord,
        'sweep_angle': sweep_angle,
        'dihedral_angle': dihedral_angle,
        'airfoil_data': airfoil_data,
        'n_span': n_span,
        'n_chord': n_chord,
        'alpha': alpha,
        'V_inf': V_inf,
        'rho': rho,
        'run_parameter_study': run_parameter_study,
        'create_plots': create_plots,
        'debug_mode': debug_mode
    }

class EnhancedWingPanel:
    """Enhanced Wing Panel Method with airfoil support and corrected aerodynamics"""
    
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        
        # Wing geometry parameters
        self.span = 0
        self.root_chord = 0
        self.tip_chord = 0
        self.sweep_angle = 0
        self.dihedral_angle = 0
        self.airfoil_data = None
        
        # Panel data
        self.panels = []
        self.control_points = []
        self.normal_vectors = []
        self.panel_areas = []
        self.panel_centers = []
        
        # Solution data
        self.source_strengths = []
        self.pressure_coefficients = []
        self.influence_matrix = None
        self.rhs_vector = None
        
        # Results
        self.CL = 0
        self.CM = 0
        self.CD_induced = 0
        self.total_lift = 0
        self.wing_area = 0
        
    def generate_wing_geometry(self, span, root_chord, tip_chord, sweep_angle, 
                             dihedral_angle, airfoil_data, n_span, n_chord):
        """Enhanced wing geometry generation with airfoil cross-sections"""
        
        print("1. Generating wing geometry with airfoil cross-sections...")
        
        self.span = span
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.sweep_angle = sweep_angle
        self.dihedral_angle = dihedral_angle
        self.airfoil_data = airfoil_data
        
        # Convert angles to radians
        sweep_rad = np.radians(sweep_angle)
        dihedral_rad = np.radians(dihedral_angle)
        
        # Get airfoil coordinates
        if airfoil_data:
            xu_norm = airfoil_data['xu']  # Upper surface x
            yu_norm = airfoil_data['yu']  # Upper surface y
            xl_norm = airfoil_data['xl']  # Lower surface x
            yl_norm = airfoil_data['yl']  # Lower surface y
        else:
            # Default to flat plate if no airfoil data
            x_flat = np.linspace(0, 1, n_chord + 1)
            xu_norm = xl_norm = x_flat
            yu_norm = yl_norm = np.zeros_like(x_flat)
        
        # Generate spanwise positions with cosine clustering
        eta = np.linspace(0, 1, n_span + 1)
        eta = 0.5 * (1 - np.cos(np.pi * eta))  # Cosine clustering
        y_positions = eta * span/2
        
        # Generate chordwise positions based on airfoil
        # Combine upper and lower surface points
        x_combined = np.concatenate([xu_norm, xl_norm[1:-1]])  # Avoid duplicate LE/TE
        y_combined = np.concatenate([yu_norm, yl_norm[1:-1]])
        
        # Sort by x-coordinate and remove duplicates
        sort_idx = np.argsort(x_combined)
        x_airfoil = x_combined[sort_idx]
        y_airfoil = y_combined[sort_idx]
        
        # Generate chordwise panel breakdown
        xi = np.linspace(0, 1, n_chord + 1)
        xi = 0.5 * (1 - np.cos(np.pi * xi))  # Cosine clustering
        
        panels = []
        control_points = []
        normal_vectors = []
        panel_areas = []
        panel_centers = []
        
        # Generate panels for right wing half
        for i in range(n_span):
            y1, y2 = y_positions[i], y_positions[i+1]
            
            # Linear taper
            c1 = root_chord - (root_chord - tip_chord) * (y1 / (span/2))
            c2 = root_chord - (root_chord - tip_chord) * (y2 / (span/2))
            
            # Sweep effect on leading edge
            x_le1 = y1 * np.tan(sweep_rad)
            x_le2 = y2 * np.tan(sweep_rad)
            
            # Dihedral effect
            z1 = y1 * np.tan(dihedral_rad)
            z2 = y2 * np.tan(dihedral_rad)
            
            # Interpolate airfoil shape at both stations
            z_airfoil_1 = np.interp(xi, x_airfoil, y_airfoil) * c1
            z_airfoil_2 = np.interp(xi, x_airfoil, y_airfoil) * c2
            
            # Generate panels along chord
            for j in range(n_chord):
                xi1, xi2 = xi[j], xi[j+1]
                
                # Panel corners with airfoil shape
                x1 = x_le1 + xi1 * c1
                x2 = x_le1 + xi2 * c1  
                x3 = x_le2 + xi2 * c2
                x4 = x_le2 + xi1 * c2
                
                # Z-coordinates from airfoil + dihedral
                z1_panel = z1 + z_airfoil_1[j]
                z2_panel = z1 + z_airfoil_1[j+1]  
                z3_panel = z2 + z_airfoil_2[j+1]
                z4_panel = z2 + z_airfoil_2[j]
                
                # Panel vertices including dihedral and airfoil shape
                p1 = np.array([x1, y1, z1_panel])
                p2 = np.array([x2, y1, z2_panel])
                p3 = np.array([x3, y2, z3_panel])
                p4 = np.array([x4, y2, z4_panel])
                
                panels.append([p1, p2, p3, p4])
                
                # Control point at panel centroid
                cp = 0.25 * (p1 + p2 + p3 + p4)
                control_points.append(cp)
                
                # Panel center
                center = 0.25 * (p1 + p2 + p3 + p4)
                panel_centers.append(center)
                
                # Calculate panel normal vector - CRITICAL FIX
                # Use proper cross product for consistent orientation
                v1 = p2 - p1  # Chordwise vector
                v2 = p4 - p1  # Spanwise vector
                normal = np.cross(v1, v2)
                normal_mag = np.linalg.norm(normal)
                
                if normal_mag > 1e-12:
                    normal = normal / normal_mag
                    # Ensure normal points generally upward 
                    # (adjust for airfoil camber but maintain consistency)
                    if np.dot(normal, [0, 0, 1]) < 0:
                        normal = -normal
                else:
                    normal = np.array([0, 0, 1])  # Default upward normal
                    
                normal_vectors.append(normal)
                
                # Calculate panel area using cross product
                diag1 = p3 - p1
                diag2 = p4 - p2
                area = 0.5 * np.linalg.norm(np.cross(diag1, diag2))
                panel_areas.append(area)
        
        # Mirror for left wing half
        n_panels_half = len(panels)
        for i in range(n_panels_half):
            panel = panels[i]
            
            # Mirror panel vertices across xz-plane
            mirrored_panel = []
            for vertex in panel:
                mirrored_vertex = np.array([vertex[0], -vertex[1], vertex[2]])
                mirrored_panel.append(mirrored_vertex)
            # Reverse order to maintain proper orientation
            mirrored_panel.reverse()
            panels.append(mirrored_panel)
            
            # Mirror control point
            cp = control_points[i]
            mirrored_cp = np.array([cp[0], -cp[1], cp[2]])
            control_points.append(mirrored_cp)
            
            # Mirror panel center
            center = panel_centers[i]
            mirrored_center = np.array([center[0], -center[1], center[2]])
            panel_centers.append(mirrored_center)
            
            # Mirror normal vector (keep upward orientation)
            normal = normal_vectors[i]
            mirrored_normal = np.array([normal[0], -normal[1], normal[2]])
            normal_vectors.append(mirrored_normal)
            
            # Same area for mirrored panel
            panel_areas.append(panel_areas[i])
        
        # Store as numpy arrays for efficient computation
        self.panels = np.array(panels)
        self.control_points = np.array(control_points)
        self.normal_vectors = np.array(normal_vectors)
        self.panel_areas = np.array(panel_areas)
        self.panel_centers = np.array(panel_centers)
        self.wing_area = np.sum(self.panel_areas)
        
        if self.debug_mode:
            print(f"   Generated {len(panels)} panels ({n_panels_half} per wing half)")
            print(f"   Total wing area: {self.wing_area:.2f} m²")
            print(f"   Average panel area: {np.mean(self.panel_areas):.4f} m²")
            print(f"   Airfoil type: {airfoil_data['type'] if airfoil_data else 'Flat plate'}")
            print(f"   Normal vector check: {np.mean([n[2] for n in self.normal_vectors]):.3f} (should be > 0)")
        
        return len(panels)
    
    def calculate_source_influence_coefficient(self, source_panel_vertices, control_point):
        """
        Calculate influence coefficient of a source panel on a control point
        This is the critical function that was causing zero lift
        """
        # Get panel corners
        p1, p2, p3, p4 = source_panel_vertices
        
        # Panel center
        panel_center = 0.25 * (p1 + p2 + p3 + p4)
        
        # Vector from panel center to control point
        r_vec = control_point - panel_center
        r_mag = np.linalg.norm(r_vec)
        
        # Avoid singularity for self-influence
        if r_mag < 1e-10:
            return 0.5  # Standard value for flat panel self-influence
        
        # Calculate panel area and normal
        v1 = p2 - p1
        v2 = p4 - p1
        panel_normal = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(panel_normal)
        
        if area < 1e-12:
            return 0.0
        
        panel_normal = panel_normal / np.linalg.norm(panel_normal)
        
        # Source influence using point source approximation
        # More sophisticated methods would use analytical integration
        influence = area / (4 * np.pi * r_mag)
        
        # The influence coefficient should give the normal component of induced velocity
        # when multiplied by source strength
        return influence / r_mag  # Additional 1/r factor for velocity calculation
    
    def build_influence_matrix(self, alpha_deg, V_inf=1.0):
        """Build the influence coefficient matrix - CRITICAL FIX"""
        
        print("Building coefficient matrix for {} panels...".format(len(self.panels)))
        
        alpha_rad = np.radians(alpha_deg)
        n_panels = len(self.panels)
        
        # Initialize matrix and RHS vector
        self.influence_matrix = np.zeros((n_panels, n_panels))
        self.rhs_vector = np.zeros(n_panels)
        
        # Freestream velocity components - CRITICAL FIX
        u_inf = V_inf * np.cos(alpha_rad)  # x-component
        v_inf = 0.0                        # y-component  
        w_inf = V_inf * np.sin(alpha_rad)  # z-component (upward for positive AoA)
        V_inf_vec = np.array([u_inf, v_inf, w_inf])
        
        if self.debug_mode:
            print(f"   Freestream velocity components: u={u_inf:.4f}, w={w_inf:.4f}")
        
        # Build influence matrix
        for i in range(n_panels):
            if i % 20 == 0:
                print(f"Processing panel {i}/{n_panels}")
            
            # Control point and normal for panel i
            cp_i = self.control_points[i]
            normal_i = self.normal_vectors[i]
            
            # Right-hand side: negative of freestream velocity normal component
            # This enforces the no-penetration boundary condition: V·n = 0
            self.rhs_vector[i] = -np.dot(V_inf_vec, normal_i)
            
            # Fill influence matrix row i
            for j in range(n_panels):
                if i == j:
                    # Self-influence coefficient for flat panel
                    self.influence_matrix[i, j] = 0.5
                else:
                    # Influence of panel j on control point i
                    source_panel = self.panels[j]
                    
                    # Get basic influence coefficient
                    influence = self.calculate_source_influence_coefficient(source_panel, cp_i)
                    
                    # Project onto normal direction of panel i
                    panel_j_center = self.panel_centers[j]
                    r_vec = cp_i - panel_j_center
                    r_mag = np.linalg.norm(r_vec)
                    
                    if r_mag > 1e-12:
                        # Normal velocity component induced by unit source on panel j
                        normal_component = np.dot(r_vec / r_mag, normal_i)
                        self.influence_matrix[i, j] = influence * normal_component
                    else:
                        self.influence_matrix[i, j] = 0.0
        
        if self.debug_mode:
            print(f"   Matrix condition number: {np.linalg.cond(self.influence_matrix):.2e}")
            print(f"   RHS vector magnitude: {np.linalg.norm(self.rhs_vector):.6f}")
            print(f"   Max RHS component: {np.max(np.abs(self.rhs_vector)):.6f}")
    
    def solve_panel_equations(self):
        """Solve the panel method equations"""
        
        print("2. Solving panel method equations...")
        
        try:
            # Solve the linear system A * sigma = b
            self.source_strengths = np.linalg.solve(self.influence_matrix, self.rhs_vector)
            
            if self.debug_mode:
                print(f"   Solution found successfully")
                print(f"   Max source strength: {np.max(np.abs(self.source_strengths)):.6f}")
                print(f"   Mean source strength: {np.mean(self.source_strengths):.6f}")
                
        except np.linalg.LinAlgError as e:
            print(f"   Warning: Linear algebra error: {e}")
            print("   Using least squares solution...")
            self.source_strengths = np.linalg.lstsq(self.influence_matrix, self.rhs_vector, rcond=None)[0]
    
    def calculate_pressure_distribution(self, alpha_deg, V_inf=1.0):
        """Calculate pressure coefficients using Bernoulli's equation"""
        
        print("3. Computing pressure distribution...")
        
        alpha_rad = np.radians(alpha_deg)
        n_panels = len(self.panels)
        
        # Freestream velocity components
        u_inf = V_inf * np.cos(alpha_rad)
        w_inf = V_inf * np.sin(alpha_rad)
        
        self.pressure_coefficients = np.zeros(n_panels)
        
        for i in range(n_panels):
            # Start with freestream velocity
            u_local = u_inf
            v_local = 0.0
            w_local = w_inf
            
            # Add induced velocities from all source panels
            cp_i = self.control_points[i]
            
            for j in range(n_panels):
                if i != j:  # Skip self-influence
                    # Vector from panel j center to control point i
                    r_vec = cp_i - self.panel_centers[j]
                    r_mag = np.linalg.norm(r_vec)
                    
                    if r_mag > 1e-12:
                        # Induced velocity from source panel j
                        sigma_j = self.source_strengths[j]
                        area_j = self.panel_areas[j]
                        
                        # Point source velocity field
                        velocity_induced = sigma_j * area_j * r_vec / (4 * np.pi * r_mag**3)
                        
                        u_local += velocity_induced[0]
                        v_local += velocity_induced[1] 
                        w_local += velocity_induced[2]
            
            # Total velocity magnitude
            V_local = np.sqrt(u_local**2 + v_local**2 + w_local**2)
            
            # Pressure coefficient from Bernoulli's equation
            if V_inf > 1e-12:
                self.pressure_coefficients[i] = 1.0 - (V_local / V_inf)**2
            else:
                self.pressure_coefficients[i] = 0.0
        
        if self.debug_mode:
            cp_stats = self.pressure_coefficients
            print(f"   Pressure coefficient range: {np.min(cp_stats):.4f} to {np.max(cp_stats):.4f}")
            print(f"   Mean pressure coefficient: {np.mean(cp_stats):.4f}")
    
    def calculate_forces_and_moments(self, alpha_deg, V_inf=1.0, rho=1.225):
        """Calculate aerodynamic forces and moments"""
        
        print("4. Calculating aerodynamic coefficients...")
        
        if len(self.pressure_coefficients) == 0:
            print("Error: No pressure distribution available")
            return
        
        # Initialize force components
        total_force = np.array([0.0, 0.0, 0.0])
        total_moment = np.array([0.0, 0.0, 0.0])
        
        # Reference point for moments (quarter chord of root chord)
        root_quarter_chord = 0.25 * self.root_chord
        ref_point = np.array([root_quarter_chord, 0.0, 0.0])
        
        # Dynamic pressure
        q_inf = 0.5 * rho * V_inf**2
        
        # Integrate forces over all panels
        for i in range(len(self.panels)):
            # Pressure and area for this panel
            cp_i = self.pressure_coefficients[i]
            area_i = self.panel_areas[i]
            normal_i = self.normal_vectors[i]
            center_i = self.panel_centers[i]
            
            # Pressure force on panel (positive inward)
            pressure_force = cp_i * q_inf * area_i
            force_vector = pressure_force * normal_i
            
            # Add to total force
            total_force += force_vector
            
            # Moment about reference point
            r_arm = center_i - ref_point
            moment_vector = np.cross(r_arm, force_vector)
            total_moment += moment_vector
        
        # Convert to aerodynamic coefficients
        if q_inf > 0 and self.wing_area > 0:
            # Force coefficients
            self.CL = total_force[2] / (q_inf * self.wing_area)  # Lift (z-direction)
            CD_pressure = -total_force[0] / (q_inf * self.wing_area)  # Drag (x-direction)
            
            # Moment coefficient (pitching moment about y-axis)
            mean_chord = self.wing_area / self.span  # Mean aerodynamic chord approximation
            self.CM = total_moment[1] / (q_inf * self.wing_area * mean_chord)
            
            # Total forces
            self.total_lift = total_force[2]
            
        else:
            self.CL = 0.0
            self.CM = 0.0
            self.total_lift = 0.0
        
        if self.debug_mode:
            print(f"   Total force vector: [{total_force[0]:.2f}, {total_force[1]:.2f}, {total_force[2]:.2f}] N")
            print(f"   Reference point: [{ref_point[0]:.2f}, {ref_point[1]:.2f}, {ref_point[2]:.2f}] m")
    
    def run_analysis(self, span, root_chord, tip_chord, sweep_angle, dihedral_angle,
                    airfoil_data, alpha, n_span=10, n_chord=12, V_inf=1.0, rho=1.225):
        """Run complete panel method analysis with airfoil"""
        
        print("Starting wing analysis with {} spanwise panels...".format(n_span))
        print(f"Wing span: {span} m")
        print(f"Root chord: {root_chord} m") 
        print(f"Airfoil: {airfoil_data['type'] if airfoil_data else 'Flat plate'}")
        print(f"Angle of attack: {alpha}°")
        print("-" * 50)
        
        # Generate geometry with airfoil
        n_panels = self.generate_wing_geometry(span, root_chord, tip_chord, 
                                             sweep_angle, dihedral_angle, airfoil_data, n_span, n_chord)
        
        # Build and solve system
        self.build_influence_matrix(alpha, V_inf)
        self.solve_panel_equations()
        
        # Calculate pressure distribution
        self.calculate_pressure_distribution(alpha, V_inf)
        
        # Calculate forces and moments
        self.calculate_forces_and_moments(alpha, V_inf, rho)
        
        # Display results
        self.display_results()
        
        return self.CL, self.CM
    
    def display_results(self):
        """Display analysis results"""
        
        print("\n" + "="*60)
        print("AERODYNAMIC ANALYSIS RESULTS")
        print("="*60)
        print(f"Lift Coefficient (CL):     {self.CL:.4f}")
        print(f"Moment Coefficient (CM):   {self.CM:.4f}")
        print(f"Total Lift Force:          {self.total_lift:.1f} N")
        print(f"Wing Area:                 {self.wing_area:.2f} m²")
        
        if self.span > 0 and self.wing_area > 0:
            aspect_ratio = self.span**2 / self.wing_area
            print(f"Aspect Ratio:              {aspect_ratio:.2f}")
            
            if self.CL != 0:
                # Rough center of pressure estimation
                cp_x = -self.CM / self.CL * (self.wing_area / self.span) + 0.25 * self.root_chord
                print(f"Center of Pressure:        {cp_x:.3f} m")
        
        print("="*60)
        
        # Validation check
        if abs(self.CL) < 1e-6:
            print("\n⚠️  WARNING: Lift coefficient is near zero!")
            print("   This suggests an issue with the implementation.")
        else:
            print(f"\n✅ Analysis appears successful (CL = {self.CL:.4f})")
    
    def parameter_study(self, base_params, study_type="sweep"):
        """Run educational parameter studies"""
        
        print("5. Creating educational visualizations...")
        print("6. Running educational parameter study...")
        print("=== EDUCATIONAL PARAMETER STUDY ===")
        
        if study_type == "sweep":
            print("Analyzing effects of sweep angle...")
            print("1. Studying sweep angle effects...")
            
            sweep_angles = [0, 15, 30, 45]
            results = []
            
            for sweep in sweep_angles:
                print(f"  Processing sweep angle: {sweep}°")
                
                # Temporarily modify parameters for study
                temp_wing = EnhancedWingPanel(debug_mode=False)
                CL, CM = temp_wing.run_analysis(
                    span=base_params['span'],
                    root_chord=base_params['root_chord'], 
                    tip_chord=base_params['tip_chord'],
                    sweep_angle=sweep,
                    dihedral_angle=base_params['dihedral_angle'],
                    airfoil_data=base_params['airfoil_data'],
                    alpha=base_params['alpha'],
                    n_span=base_params['n_span'],
                    n_chord=base_params['n_chord']
                )
                
                results.append((sweep, CL, CM))
                print(f"  Sweep {sweep}°: CL = {CL:.4f}")
            
            return results
    
    def create_visualization(self):
        """Create 3D visualization of the wing and results"""
        
        if len(self.panels) == 0:
            print("No geometry to visualize")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot wing panels
        for i, panel in enumerate(self.panels):
            # Panel vertices
            vertices = np.array([panel[0], panel[1], panel[2], panel[3], panel[0]])
            
            # Color based on pressure coefficient
            if i < len(self.pressure_coefficients):
                cp = self.pressure_coefficients[i]
                color = plt.cm.RdBu_r(0.5 + cp * 0.5)  # Blue for low pressure, red for high
            else:
                color = 'lightblue'
            
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'k-', alpha=0.3)
            
            # Fill panel with color
            if i < len(self.pressure_coefficients):
                ax.plot_trisurf(panel[:, 0], panel[:, 1], panel[:, 2], color=color, alpha=0.6)
        
        # Plot control points
        if len(self.control_points) > 0:
            cp_array = np.array(self.control_points)
            ax.scatter(cp_array[:, 0], cp_array[:, 1], cp_array[:, 2], 
                      c='red', s=2, alpha=0.8, label='Control Points')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)') 
        ax.set_zlabel('Z (m)')
        ax.set_title('Wing Panel Method Analysis\nCL = {:.4f}, CM = {:.4f}'.format(self.CL, self.CM))
        ax.legend()
        
        # Equal aspect ratio
        max_range = max([
            np.max(cp_array[:, 0]) - np.min(cp_array[:, 0]),
            np.max(cp_array[:, 1]) - np.min(cp_array[:, 1])
        ]) * 0.6
        
        mid_x = np.mean(cp_array[:, 0])
        mid_y = np.mean(cp_array[:, 1])
        mid_z = np.mean(cp_array[:, 2])
        
        ax.set_xlim([mid_x - max_range/2, mid_x + max_range/2])
        ax.set_ylim([mid_y - max_range/2, mid_y + max_range/2])
        ax.set_zlim([mid_z - max_range/4, mid_z + max_range/4])
        
        plt.tight_layout()
        plt.show()

def quick_test_mode():
    """Quick test mode without user interaction for debugging"""
    print("Running in quick test mode...")
    
    # Quick test parameters
    test_params = {
        'span': 10.606,
        'root_chord': 8.02,
        'tip_chord': 8.02 * 0.6,
        'sweep_angle': 0,
        'dihedral_angle': 0,
        'airfoil_data': {'xu': np.linspace(0,1,50), 'yu': np.zeros(50), 
                        'xl': np.linspace(0,1,50), 'yl': np.zeros(50),
                        'type': 'Flat Plate (Test)'},
        'alpha': 4,
        'n_span': 8,
        'n_chord': 10,
        'V_inf': 1.0,
        'rho': 1.225
    }
    
    # Run analysis
    wing = EnhancedWingPanel(debug_mode=True)
    CL, CM = wing.run_analysis(**test_params)
    
    return wing, CL, CM

def main():
    """Main analysis function with user inputs"""
    
    # Get user inputs
    user_inputs = get_user_inputs()
    
    # Initialize wing analysis
    wing = EnhancedWingPanel(debug_mode=user_inputs['debug_mode'])
    
    # Run main analysis
    CL, CM = wing.run_analysis(
        span=user_inputs['span'],
        root_chord=user_inputs['root_chord'],
        tip_chord=user_inputs['tip_chord'],
        sweep_angle=user_inputs['sweep_angle'],
        dihedral_angle=user_inputs['dihedral_angle'],
        airfoil_data=user_inputs['airfoil_data'],
        alpha=user_inputs['alpha'],
        n_span=user_inputs['n_span'],
        n_chord=user_inputs['n_chord'],
        V_inf=user_inputs['V_inf'],
        rho=user_inputs['rho']
    )
    
    # Run parameter studies if requested
    if user_inputs['run_parameter_study']:
        results = wing.parameter_study(user_inputs, "sweep")
    
    # Create visualizations if requested
    if user_inputs['create_plots']:
        try:
            wing.create_visualization()
        except Exception as e:
            print(f"Visualization error: {e}")
    
    print("\nAnalysis completed successfully!")
    return wing, CL, CM

if __name__ == "__main__":
    import sys
    
    # Check if running in interactive mode or quick test
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick test mode for debugging
        wing_analysis, lift_coeff, moment_coeff = quick_test_mode()
        print(f"\nQUICK TEST RESULTS:")
        print(f"CL = {lift_coeff:.4f}")
        print(f"CM = {moment_coeff:.4f}")
    else:
        # Interactive mode with full user inputs
        try:
            wing_analysis, lift_coeff, moment_coeff = main()
        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted by user.")
        except Exception as e:
            print(f"\nError during analysis: {e}")
            print("Running quick test instead...")
            wing_analysis, lift_coeff, moment_coeff = quick_test_mode()
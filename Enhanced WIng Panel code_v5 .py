# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:42:27 2025

@author: hknar
"""
"""
Wing Panel Method Implementation - Educational Version
Original concept and structure by AeroGuruHK (https://aeroguruhk.github.io/hk/videos.html)
Code debugging and optimization assistance provided by Claude (Anthropic)

This implementation uses a vortex ring panel method to analyze wing aerodynamics.
Educational features include:
- Lift and moment coefficient calculations
- Pressure distribution visualization
- Span loading analysis
- Wing efficiency parameters
- Interactive parameter studies
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------ Panel Method Core ------------------

def wing_panel_code(airfoil, planform, nb, rootchord, aoa):
    vinf = np.array([np.cos(np.radians(aoa)), 0, np.sin(np.radians(aoa))])
    xpbase = airfoil['x']
    zpbase = airfoil['z']
    b = planform['span']
    sweep = planform['sweep']
    dihedral = planform['dihedral']
    twist = planform['twist']
    taper = planform['taper']

    xp = xpbase * rootchord
    zp = zpbase * rootchord

    plt.figure(figsize=(8, 6))
    plt.axis('equal')
    plt.plot(xp, zp, 'b-', linewidth=2)
    plt.title("Airfoil Shape")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.grid(True)
    plt.show()

    r, rc, nw, sw, se, ne, no, we, so, ea, wp, bp, extra1 = wing(
        xp, zp, b, nb, vinf, sweep, dihedral, twist, taper, rootchord
    )

    ac = 0.5 * v_cross(r[:, sw] - r[:, ne], r[:, se] - r[:, nw])
    nc = ac / v_mag(ac)

    npanels = len(bp)  # Use bp length as the number of panels
    coef = np.zeros((npanels, npanels))

    print(f"Building coefficient matrix for {npanels} panels...")
    
    for nn in range(len(bp)):
        if nn % 20 == 0:  # Progress indicator
            print(f"Processing panel {nn}/{len(bp)}")
            
        n = bp[nn]
        if n < npanels:  # Ensure we don't exceed matrix bounds
            # Calculate influence of all panels on panel n
            for m in range(len(bp)):  # Loop over all source panels
                m_idx = bp[m]
                if m_idx < len(nw):
                    # Vortex ring: 4 filaments forming a closed loop
                    # Filament 1: nw -> sw
                    cmn1 = ffil(rc[:, n:n+1], r[:, nw[m_idx:m_idx+1]], r[:, sw[m_idx:m_idx+1]])
                    # Filament 2: sw -> se  
                    cmn2 = ffil(rc[:, n:n+1], r[:, sw[m_idx:m_idx+1]], r[:, se[m_idx:m_idx+1]])
                    # Filament 3: se -> ne
                    cmn3 = ffil(rc[:, n:n+1], r[:, se[m_idx:m_idx+1]], r[:, ne[m_idx:m_idx+1]])
                    # Filament 4: ne -> nw
                    cmn4 = ffil(rc[:, n:n+1], r[:, ne[m_idx:m_idx+1]], r[:, nw[m_idx:m_idx+1]])
                    
                    # Total influence from panel m
                    cmn_panel = cmn1.flatten() + cmn2.flatten() + cmn3.flatten() + cmn4.flatten()
                    
                    # Normal component at control point n
                    coef[n, m] = np.dot(nc[:, n], cmn_panel)

    return rc, r, nc, bp, wp, ea, we, nw, sw, se, ne, no, so, coef, vinf

def v_cross(a, b):
    return np.cross(a.T, b.T).T

def v_mag(v):
    return np.linalg.norm(v, axis=0)

def ffil(rc, p1, p2):
    """
    Calculate the induced velocity at point rc due to a vortex filament from p1 to p2
    Using the Biot-Savart law
    """
    # Handle the case where rc is a single point and p1, p2 are arrays of points
    if len(rc.shape) == 1:
        rc = rc.reshape(-1, 1)
    
    # Ensure p1 and p2 are 2D arrays
    if len(p1.shape) == 1:
        p1 = p1.reshape(-1, 1)
    if len(p2.shape) == 1:
        p2 = p2.reshape(-1, 1)
    
    # Number of filaments
    n_filaments = p1.shape[1]
    
    # Initialize velocity array
    velocity = np.zeros((3, n_filaments))
    
    for i in range(n_filaments):
        # Vector from p1 to p2 (filament direction)
        dl = p2[:, i] - p1[:, i]
        
        # Vectors from filament endpoints to field point
        r1 = rc.flatten() - p1[:, i]
        r2 = rc.flatten() - p2[:, i]
        
        # Cross products
        r1_cross_r2 = np.cross(r1, r2)
        r1_cross_dl = np.cross(r1, dl)
        
        # Magnitudes
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        r1_cross_r2_mag_sq = np.dot(r1_cross_r2, r1_cross_r2)
        
        # Avoid singularities
        if r1_mag < 1e-10 or r2_mag < 1e-10 or r1_cross_r2_mag_sq < 1e-10:
            velocity[:, i] = 0.0
            continue
        
        # Biot-Savart law (without 4π factor since we'll use unit circulation)
        factor = (np.dot(dl, r1)/r1_mag - np.dot(dl, r2)/r2_mag) / r1_cross_r2_mag_sq
        velocity[:, i] = factor * r1_cross_r2
    
    return velocity

def wing(xp, zp, span, nb, vinf, sweep, dihedral, twist, taper, rootchord):
    nc = len(xp)
    half_span = span / 2
    dy = half_span / nb

    # Total number of points
    total_points = nc * (nb + 1)  # +1 because we need points at both edges
    r = np.zeros((3, total_points))
    
    nw, sw, se, ne = [], [], [], []
    no, we, so, ea = [], [], [], []
    wp, bp = [], []

    # Generate points for all spanwise stations (including tip)
    for j in range(nb + 1):
        y = j * dy
        chord = rootchord * (1 - (1 - taper) * (y / half_span))
        sweep_offset = y * np.tan(np.radians(sweep))
        dihedral_offset = y * np.tan(np.radians(dihedral))
        twist_angle = np.radians(twist * (y / half_span))

        x_twisted = xp * chord
        z_twisted = zp * chord
        x_rot = x_twisted * np.cos(twist_angle) + z_twisted * np.sin(twist_angle)
        z_rot = -x_twisted * np.sin(twist_angle) + z_twisted * np.cos(twist_angle)

        for i in range(nc):
            idx = j * nc + i
            r[:, idx] = [x_rot[i] + sweep_offset, y, z_rot[i] + dihedral_offset]

    # Generate panels (only for interior sections)
    for j in range(nb):
        for i in range(nc - 1):  # nc-1 because we create panels between points
            panel_idx = j * (nc - 1) + i
            
            # Panel corner indices
            nw.append(j * nc + i)
            sw.append((j + 1) * nc + i)
            se.append((j + 1) * nc + i + 1)
            ne.append(j * nc + i + 1)
            
            # Panel center indices (for now, same as nw)
            no.append(panel_idx)
            we.append(panel_idx)
            so.append(panel_idx)
            ea.append(panel_idx)
            
            bp.append(panel_idx)
            
            # Wake panels (at the trailing edge)
            if j == nb - 1:  # Last spanwise section
                wp.append((j + 1) * nc + i)

    # Convert to integer arrays
    nw = np.array(nw, dtype=int)
    sw = np.array(sw, dtype=int)
    se = np.array(se, dtype=int)
    ne = np.array(ne, dtype=int)
    no = np.array(no, dtype=int)
    we = np.array(we, dtype=int)
    so = np.array(so, dtype=int)
    ea = np.array(ea, dtype=int)
    wp = np.array(wp, dtype=int)
    bp = np.array(bp, dtype=int)

    # Create control points (panel centers)
    npanels = len(bp)
    rc = np.zeros((3, npanels))
    for i in range(npanels):
        # Panel center as average of corner points
        rc[:, i] = 0.25 * (r[:, nw[i]] + r[:, sw[i]] + r[:, se[i]] + r[:, ne[i]])

    # Visualize mesh
    plot_wing_mesh(r, nw, sw, se, ne)

    return r, rc, nw, sw, se, ne, no, we, so, ea, wp, bp, np.zeros((3, npanels))

# ------------------ Solver Functions ------------------

def solve_panel_strengths(coef, nc, vinf, wp, bp):
    npanels = len(bp)  # Number of panels should match bp length
    
    # Create right-hand side vector
    rm = -np.dot(nc.T, vinf)
    
    # Ensure wp indices are within bounds and set wake panels to zero
    valid_wp = wp[wp < len(rm)]
    if len(valid_wp) > 0:
        rm[valid_wp] = 0
    
    # The coefficient matrix should already be square (npanels x npanels)
    # If it's not, we need to resize it
    if coef.shape[0] != coef.shape[1]:
        print(f"Warning: Coefficient matrix is {coef.shape}, making it square...")
        min_dim = min(coef.shape[0], coef.shape[1])
        coef = coef[:min_dim, :min_dim]
        rm = rm[:min_dim]
        bp = bp[bp < min_dim]
    
    # Add regularization to improve conditioning
    regularization = 1e-12
    np.fill_diagonal(coef, coef.diagonal() + regularization)
    
    # For Kutta condition, replace last equation with sum of trailing edge panel circulations = 0
    if len(bp) > 0:
        coef[-1, :] = 0
        coef[-1, bp] = 1  # Sum of all panel circulations = 0 (simplified Kutta condition)
        rm[-1] = 0
    
    # Check matrix condition number
    cond_num = np.linalg.cond(coef)
    if cond_num > 1e12:
        print(f"Warning: Matrix is poorly conditioned (condition number: {cond_num:.2e})")
    
    # Solve system
    try:
        ga = np.linalg.solve(coef, rm)
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error: {e}")
        print(f"Coefficient matrix shape: {coef.shape}")
        print(f"RHS vector shape: {rm.shape}")
        print(f"Matrix condition number: {cond_num:.2e}")
        # Use least squares as fallback with regularization
        print("Using least squares solution with regularization...")
        ga = np.linalg.lstsq(coef, rm, rcond=1e-10)[0]
    
    return ga

def compute_velocity_pressure(rc, r, vinf, ga, nc, we, no, so, ea, bp, nw, sw, se, ne):
    npanels = rc.shape[1]
    ga3 = np.tile(ga[:npanels], (3, 1))  # Only use panel strengths
    v = np.zeros((3, npanels))

    for n in range(npanels):
        cmn = (ffil(rc[:, n:n+1], r[:, nw[n:n+1]], r[:, sw[n:n+1]]) +
               ffil(rc[:, n:n+1], r[:, sw[n:n+1]], r[:, se[n:n+1]]) +
               ffil(rc[:, n:n+1], r[:, se[n:n+1]], r[:, ne[n:n+1]]) +
               ffil(rc[:, n:n+1], r[:, ne[n:n+1]], r[:, nw[n:n+1]]))
        v[:, n] = vinf.reshape(-1, 1).flatten() + np.sum(ga3 * cmn, axis=1)

    # Simplified pressure coefficient calculation
    v_mag_squared = np.sum(v ** 2, axis=0)
    vinf_mag_squared = np.dot(vinf, vinf)
    cp = 1 - v_mag_squared / vinf_mag_squared
    
    return cp, v

def calculate_aerodynamic_coefficients(rc, r, cp, nc, planform, vinf, rootchord, rho=1.225):
    """
    Calculate lift coefficient, moment coefficient, and other aerodynamic parameters
    """
    # Wing geometry
    span = planform['span']
    wing_area = rootchord * span * (1 + planform['taper']) / 2  # Trapezoidal wing area
    
    # Dynamic pressure
    V = np.linalg.norm(vinf)
    q = 0.5 * rho * V**2
    
    # Panel areas (simplified as rectangular panels)
    npanels = rc.shape[1]
    panel_areas = np.zeros(npanels)
    
    # Calculate panel dimensions
    dy = span / (2 * np.sqrt(npanels))  # Approximate spanwise spacing
    
    for i in range(npanels):
        # Local chord at this spanwise position
        y_local = abs(rc[1, i])
        local_chord = rootchord * (1 - (1 - planform['taper']) * (y_local / (span/2)))
        dx = local_chord / np.sqrt(npanels)  # Approximate chordwise spacing
        panel_areas[i] = dx * dy
    
    # Forces on each panel
    panel_forces = -cp * q * panel_areas  # Negative because cp is typically negative for lift
    
    # Lift calculation
    lift_total = np.sum(panel_forces * nc[2, :])  # Z-component of normal force
    CL = lift_total / (q * wing_area)
    
    # Pitching moment about quarter chord
    moment_ref = np.array([0.25 * rootchord, 0, 0])  # Quarter chord of root
    moment_total = 0
    
    for i in range(npanels):
        # Moment arm from reference point to panel center
        arm = rc[:, i] - moment_ref
        # Moment contribution (force × arm, taking y-component)
        moment_contrib = panel_forces[i] * arm[0] * nc[2, i]
        moment_total += moment_contrib
    
    CM = moment_total / (q * wing_area * rootchord)
    
    # Span efficiency factor (simplified Oswald efficiency)
    # This is approximate - real calculation requires induced drag
    AR = span**2 / wing_area  # Aspect ratio
    e_oswald = 0.85  # Typical value
    
    # Center of pressure
    if lift_total != 0:
        x_cp = moment_ref[0] - moment_total / lift_total
    else:
        x_cp = 0.25 * rootchord
    
    results = {
        'CL': CL,
        'CM': CM,
        'wing_area': wing_area,
        'aspect_ratio': AR,
        'oswald_efficiency': e_oswald,
        'lift_total': lift_total,
        'moment_total': moment_total,
        'center_of_pressure': x_cp,
        'dynamic_pressure': q,
        'panel_forces': panel_forces,
        'panel_areas': panel_areas
    }
    
    return results

def plot_educational_results(rc, r, cp, results, planform, nw, sw, se, ne):
    """
    Create comprehensive educational plots
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D Wing with Pressure Distribution
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Plot wing panels colored by pressure coefficient
    for i in range(len(nw)):
        if i < len(cp):
            x_panel = [r[0, nw[i]], r[0, sw[i]], r[0, se[i]], r[0, ne[i]], r[0, nw[i]]]
            y_panel = [r[1, nw[i]], r[1, sw[i]], r[1, se[i]], r[1, ne[i]], r[1, nw[i]]]
            z_panel = [r[2, nw[i]], r[2, sw[i]], r[2, se[i]], r[2, ne[i]], r[2, nw[i]]]
            
            # Color based on Cp value
            color_val = plt.cm.RdBu_r((cp[i] + 2) / 4)  # Normalize for color map
            ax1.plot(x_panel, y_panel, z_panel, color=color_val, linewidth=1.5)
    
    ax1.set_xlabel('X (Chordwise)')
    ax1.set_ylabel('Y (Spanwise)')
    ax1.set_zlabel('Z (Vertical)')
    ax1.set_title('3D Wing Geometry\n(Colored by Cp)')
    
    # 2. Chordwise Pressure Distribution
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Group by chordwise position (approximate)
    x_positions = rc[0, :]
    ax2.scatter(x_positions, cp, c=cp, cmap='RdBu_r', alpha=0.7)
    ax2.set_xlabel('Chordwise Position (x)')
    ax2.set_ylabel('Pressure Coefficient (Cp)')
    ax2.set_title('Chordwise Pressure Distribution')
    ax2.grid(True)
    ax2.invert_yaxis()  # Negative Cp at top (conventional)
    
    # 3. Spanwise Load Distribution
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Group by spanwise position
    y_positions = abs(rc[1, :])  # Absolute value for half-span
    span_forces = results['panel_forces']
    
    ax3.scatter(y_positions, span_forces, alpha=0.7, color='red')
    ax3.set_xlabel('Spanwise Position (y)')
    ax3.set_ylabel('Panel Force (N)')
    ax3.set_title('Spanwise Load Distribution')
    ax3.grid(True)
    
    # 4. Aerodynamic Coefficients Summary
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    
    # Create text summary
    summary_text = f"""
AERODYNAMIC RESULTS

Lift Coefficient (CL): {results['CL']:.4f}
Moment Coefficient (CM): {results['CM']:.4f}

Wing Parameters:
• Wing Area: {results['wing_area']:.2f} m²
• Aspect Ratio: {results['aspect_ratio']:.2f}
• Span: {planform['span']:.2f} m
• Root Chord: {results['wing_area']/planform['span']*(1+planform['taper'])/2:.2f} m

Performance:
• Total Lift: {results['lift_total']:.1f} N
• Center of Pressure: {results['center_of_pressure']:.3f} m
• Dynamic Pressure: {results['dynamic_pressure']:.1f} Pa

Geometry:
• Sweep: {planform['sweep']:.1f}°
• Taper Ratio: {planform['taper']:.3f}
• Twist: {planform['twist']:.1f}°
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 5. Lift Curve Theory vs Panel Method
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Theoretical lift curve slope
    AR = results['aspect_ratio']
    a0 = 2 * np.pi  # 2D lift curve slope
    CL_alpha_theory = a0 / (1 + a0 / (np.pi * AR))  # 3D lift curve slope
    
    # Show comparison (this would be extended for multiple AoA)
    aoa_range = np.linspace(-5, 15, 50)
    CL_theory = CL_alpha_theory * np.radians(aoa_range)
    
    ax5.plot(aoa_range, CL_theory, 'b-', label='Lifting Line Theory', linewidth=2)
    ax5.plot(4, results['CL'], 'ro', markersize=8, label=f'Panel Method (α=4°)')
    ax5.set_xlabel('Angle of Attack (degrees)')
    ax5.set_ylabel('Lift Coefficient (CL)')
    ax5.set_title('Lift Curve Comparison')
    ax5.grid(True)
    ax5.legend()
    ax5.set_xlim(-5, 15)
    
    # 6. Wing Efficiency Visualization
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Elliptical vs actual load distribution (simplified)
    y_ellip = np.linspace(0, planform['span']/2, 100)
    elliptical_load = np.sqrt(1 - (2*y_ellip/planform['span'])**2)
    
    ax6.plot(y_ellip, elliptical_load, 'b--', label='Elliptical (Ideal)', linewidth=2)
    
    # Actual load (simplified representation)
    y_actual = np.linspace(0, planform['span']/2, len(span_forces)//2)
    if len(y_actual) > 0:
        actual_load_normalized = abs(span_forces[:len(y_actual)]) / max(abs(span_forces))
        ax6.plot(y_actual, actual_load_normalized, 'r-', label='Panel Method', linewidth=2)
    
    ax6.set_xlabel('Spanwise Position (m)')
    ax6.set_ylabel('Normalized Load')
    ax6.set_title('Load Distribution Comparison')
    ax6.grid(True)
    ax6.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

def educational_parameter_study(airfoil, base_planform, nb, rootchord, rho=1.225):
    """
    Study effect of key parameters for educational purposes
    """
    print("=== EDUCATIONAL PARAMETER STUDY ===")
    print("Analyzing effects of sweep angle...")
    
    # Parameter ranges (reduced for speed)
    sweep_angles = [0, 15, 30, 45]
    
    results_sweep = []
    
    # 1. Sweep angle study
    print("\n1. Studying sweep angle effects...")
    for sweep in sweep_angles:
        planform = base_planform.copy()
        planform['sweep'] = sweep
        
        try:
            print(f"  Processing sweep angle: {sweep}°")
            rc, r, nc, bp, wp, ea, we, nw, sw, se, ne, no, so, coef, vinf = wing_panel_code(
                airfoil, planform, nb, rootchord, 4
            )
            ga = solve_panel_strengths(coef, nc, vinf, wp, bp)
            cp, velocity = compute_velocity_pressure(rc, r, vinf, ga, nc, we, no, so, ea, bp, nw, sw, se, ne)
            aero_results = calculate_aerodynamic_coefficients(rc, r, cp, nc, planform, vinf, rootchord, rho)
            
            results_sweep.append({
                'sweep': sweep,
                'CL': aero_results['CL'],
                'CM': aero_results['CM'],
                'AR': aero_results['aspect_ratio']
            })
            print(f"  Sweep {sweep}°: CL = {aero_results['CL']:.4f}")
        except Exception as e:
            print(f"  Sweep {sweep}°: Calculation failed - {e}")
    
    # Create educational plots for parameter study
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot sweep effects
    if results_sweep:
        sweeps = [r['sweep'] for r in results_sweep]
        CLs_sweep = [r['CL'] for r in results_sweep]
        ax1.plot(sweeps, CLs_sweep, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Sweep Angle (degrees)')
        ax1.set_ylabel('Lift Coefficient (CL)')
        ax1.set_title('Effect of Sweep Angle')
        ax1.grid(True)
    
    # Add educational annotations
    ax2.text(0.1, 0.9, "Key Learning Points:", transform=ax2.transAxes, fontsize=12, weight='bold')
    ax2.text(0.1, 0.8, "• Sweep reduces effective aspect ratio", transform=ax2.transAxes, fontsize=10)
    ax2.text(0.1, 0.7, "• Lower lift coefficient at same AoA", transform=ax2.transAxes, fontsize=10)
    ax2.text(0.1, 0.6, "• Better transonic performance", transform=ax2.transAxes, fontsize=10)
    ax2.text(0.1, 0.5, "• Affects stall characteristics", transform=ax2.transAxes, fontsize=10)
    
    ax2.text(0.1, 0.3, "Panel Method Limitations:", transform=ax2.transAxes, fontsize=12, weight='bold')
    ax2.text(0.1, 0.2, "• Linear potential flow theory", transform=ax2.transAxes, fontsize=10)
    ax2.text(0.1, 0.1, "• No viscous effects", transform=ax2.transAxes, fontsize=10)
    ax2.text(0.1, 0.0, "• No stall prediction", transform=ax2.transAxes, fontsize=10)
    
    ax2.axis('off')
    
    # Theoretical comparison
    ax3.text(0.5, 0.9, "Theoretical Relationships", transform=ax3.transAxes, 
             fontsize=12, weight='bold', ha='center')
    
    theory_text = """
Lifting Line Theory:
CL_α = a₀/(1 + a₀/(π·AR))

where:
• a₀ = 2π (2D lift slope)
• AR = b²/S (aspect ratio)

Sweep Effects:
AR_eff = AR · cos²(Λ)
CL_swept = CL_unswept · cos²(Λ)
    """
    
    ax3.text(0.1, 0.7, theory_text, transform=ax3.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top')
    ax3.axis('off')
    
    # Practical applications
    ax4.text(0.5, 0.9, "Real-World Applications", transform=ax4.transAxes, 
             fontsize=12, weight='bold', ha='center')
    
    applications_text = """
High Sweep (45°+):
• Commercial airliners
• Supersonic aircraft
• High-speed cruise

Moderate Sweep (20-40°):
• Business jets  
• Regional aircraft
• Military fighters

Low/No Sweep (0-15°):
• Light aircraft
• Gliders
• UAVs
    """
    
    ax4.text(0.1, 0.7, applications_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results_sweep

def plot_wing_mesh(r, nw, sw, se, ne):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(nw)):
        x = [r[0, nw[i]], r[0, sw[i]], r[0, se[i]], r[0, ne[i]], r[0, nw[i]]]
        y = [r[1, nw[i]], r[1, sw[i]], r[1, se[i]], r[1, ne[i]], r[1, nw[i]]]
        z = [r[2, nw[i]], r[2, sw[i]], r[2, se[i]], r[2, ne[i]], r[2, nw[i]]]
        ax.plot(x, y, z, color='blue', linewidth=1)
    ax.set_xlabel('X (Chordwise)')
    ax.set_ylabel('Y (Spanwise)') 
    ax.set_zlabel('Z (Vertical)')
    ax.set_title('Wing Panel Mesh')
    plt.show()

# ------------------ Driver ------------------

# ------------------ Main Driver Code ------------------

if __name__ == "__main__":
    # Airfoil definition (simplified NACA-like)
    airfoil = {
        'x': np.array([1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
        'z': np.array([0.0, 0.009, 0.025, 0.032, 0.029, 0.019, 0.013, 0.0, -0.006, -0.007, -0.008, -0.006, -0.001, 0.0, 0.0])
    }

    # Wing planform parameters
    planform = {
        'span': 10.606,
        'sweep': 45,
        'dihedral': 0,
        'twist': -2.0,
        'taper': 0.175
    }

    # Analysis parameters
    nb = 10  # Number of spanwise panels (reduced for speed)
    rootchord = 8.02
    aoa = 4  # Angle of attack in degrees

    print(f"Starting wing analysis with {nb} spanwise panels...")
    print(f"Wing span: {planform['span']} m")
    print(f"Root chord: {rootchord} m")
    print(f"Angle of attack: {aoa}°")
    print("-" * 50)

    try:
        # Run panel method
        print("1. Generating wing geometry and panels...")
        rc, r, nc, bp, wp, ea, we, nw, sw, se, ne, no, so, coef, vinf = wing_panel_code(
            airfoil, planform, nb, rootchord, aoa
        )

        # Solve for panel strengths
        print("2. Solving panel method equations...")
        ga = solve_panel_strengths(coef, nc, vinf, wp, bp)
        
        # Compute velocities and pressure coefficients
        print("3. Computing pressure distribution...")
        cp, velocity = compute_velocity_pressure(rc, r, vinf, ga, nc, we, no, so, ea, bp, nw, sw, se, ne)

        # Calculate aerodynamic coefficients
        print("4. Calculating aerodynamic coefficients...")
        aero_results = calculate_aerodynamic_coefficients(rc, r, cp, nc, planform, vinf, rootchord)

        # Display results
        print("\n" + "="*60)
        print("AERODYNAMIC ANALYSIS RESULTS")
        print("="*60)
        print(f"Lift Coefficient (CL):     {aero_results['CL']:.4f}")
        print(f"Moment Coefficient (CM):   {aero_results['CM']:.4f}")
        print(f"Total Lift Force:          {aero_results['lift_total']:.1f} N")
        print(f"Wing Area:                 {aero_results['wing_area']:.2f} m²")
        print(f"Aspect Ratio:              {aero_results['aspect_ratio']:.2f}")
        print(f"Center of Pressure:        {aero_results['center_of_pressure']:.3f} m")
        print("="*60)

        # Create educational visualizations
        print("\n5. Creating educational visualizations...")
        plot_educational_results(rc, r, cp, aero_results, planform, nw, sw, se, ne)

        # Run parameter study
        print("\n6. Running educational parameter study...")
        parameter_results = educational_parameter_study(airfoil, planform, nb, rootchord)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your inputs and try again.")

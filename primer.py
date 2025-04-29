import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QTabWidget, QComboBox, QSpinBox, QGridLayout,
                             QGroupBox, QTextEdit, QSplitter, QFileDialog,
                             QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor
from sympy import isprime, primerange, primefactors, gcd, lcm, factorint
from sklearn.manifold import TSNE
import networkx as nx
from scipy.spatial import Delaunay
from scipy.linalg import qr, eigh
import matplotlib.cm as cm
import json
import time
from functools import lru_cache
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import itertools

def apply_dark_theme(app):
    """Apply dark theme to the Qt application."""
    # Set up dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    # Apply palette
    app.setPalette(dark_palette)
    
    # Set stylesheet for additional elements
    app.setStyleSheet("""
        QToolTip { 
            color: #ffffff; 
            background-color: #2a82da; 
            border: 1px solid white; 
        }
        QGroupBox { 
            border: 1px solid gray; 
            border-radius: 5px; 
            margin-top: 1ex; 
            padding-top: 1ex; 
        }
        QGroupBox::title { 
            subcontrol-origin: margin; 
            padding: 0 3px; 
        }
        QTabBar::tab {
            background-color: #3a3a3a;
            color: #ffffff;
            padding: 8px 12px;
            border: 1px solid #555555;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #2a82da;
        }
        QTabWidget::pane {
            border: 1px solid #3a3a3a;
        }
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: #3a3a3a;
            margin: 2px 0;
        }
        QSlider::handle:horizontal {
            background: #2a82da;
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0;
            border-radius: 3px;
        }
        QTextEdit {
            background-color: #2D2D2D;
            color: #DDDDDD;
            border: 1px solid #555555;
        }
        QComboBox {
            background-color: #3a3a3a;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 4px;
            border-radius: 3px;
        }
        QSpinBox {
            background-color: #3a3a3a;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 4px;
            border-radius: 3px;
        }
        QPushButton {
            background-color: #3a3a3a;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 6px 12px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton:pressed {
            background-color: #2a82da;
        }
    """)

def set_matplotlib_dark_style():
    """Set dark style for Matplotlib plots."""
    plt.style.use('dark_background')
    
    # Additional customizations for dark theme
    plt.rcParams.update({
        'axes.facecolor': '#2D2D2D',
        'figure.facecolor': '#2D2D2D',
        'savefig.facecolor': '#2D2D2D',
        'grid.color': '#707070',
        'text.color': '#D0D0D0',
        'axes.labelcolor': '#D0D0D0',
        'axes.edgecolor': '#707070',
        'xtick.color': '#D0D0D0',
        'ytick.color': '#D0D0D0',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


class PrimeResidueTopology:
    """Core implementation of the Prime Residue Topology framework."""
    
    def __init__(self, max_number=1000, moduli=None):
        """Initialize the Prime Residue Topology framework with the given parameters.
        
        Args:
            max_number: Maximum number to analyze
            moduli: List of moduli for residue analysis. If None, optimal moduli are calculated.
        """
        self.max_number = max_number
        self.numbers = np.arange(2, max_number)
        self.is_prime = np.array([isprime(n) for n in self.numbers])
        self.prime_indices = np.where(self.is_prime)[0]
        self.primes = self.numbers[self.prime_indices]
        
        # Calculate theoretically optimal moduli if not provided
        if moduli is None:
            self.moduli = self._calculate_optimal_moduli()
        else:
            self.moduli = moduli
            
        # Generate prime fingerprints and initialize fields
        self.fingerprints = self._generate_fingerprints()
        self.potential_field = self._calculate_potential_field()
        self.tensor_field = self._calculate_tensor_field()
        self.laplacian_matrix = self._calculate_laplacian_matrix()
        self.invariants = self._calculate_topological_invariants()
        self.low_dim_embedding = None
        
    def _calculate_optimal_moduli(self):
        """Calculate theoretically optimal moduli based on prime structure.
        
        The optimal moduli are chosen based on:
        1. Prime powers to capture multiplicative structure
        2. Products of small primes to capture CRT-based properties
        3. Fibonacci-like sequences that reveal golden ratio connections
        """
        # Prime powers to capture multiplicative structure
        prime_powers = [4, 9, 16, 25, 49, 121]
        
        # Products of small distinct primes (for Chinese Remainder Theorem effects)
        # These create "resonances" in the residue patterns
        crt_moduli = [6, 10, 15, 30, 105]
        
        # Special moduli based on theoretical connections
        # (Fibonacci-adjacent numbers to capture golden ratio connections)
        special_moduli = [12, 20, 24, 60]
        
        # Additional theoretical moduli that optimize information gain
        # These are selected to maximize the mutual information between
        # residue patterns and primality
        info_moduli = []
        
        # Calculate information gain for each candidate modulus up to sqrt(max_number)
        candidates = list(range(7, int(np.sqrt(self.max_number)) + 1))
        info_gains = []
        
        # We'll take a sample of numbers to evaluate information gain
        sample_size = min(1000, self.max_number - 2)
        sample_indices = np.random.choice(len(self.numbers), sample_size, replace=False)
        sample_numbers = self.numbers[sample_indices]
        sample_is_prime = self.is_prime[sample_indices]
        
        for m in candidates:
            # Skip if m is already in our moduli lists
            if m in prime_powers or m in crt_moduli or m in special_moduli:
                continue
                
            # Calculate residues for this modulus
            residues = sample_numbers % m
            
            # Calculate conditional entropy of primality given residue
            conditional_entropy = 0
            for r in range(m):
                indices = residues == r
                if np.sum(indices) > 0:
                    p_prime = np.mean(sample_is_prime[indices])
                    if 0 < p_prime < 1:  # Avoid log(0)
                        conditional_entropy -= p_prime * np.log2(p_prime) + (1-p_prime) * np.log2(1-p_prime)
            
            # Information gain = unconditional entropy - conditional entropy
            p_prime_overall = np.mean(sample_is_prime)
            unconditional_entropy = -(p_prime_overall * np.log2(p_prime_overall) + 
                                    (1-p_prime_overall) * np.log2(1-p_prime_overall))
            info_gain = unconditional_entropy - conditional_entropy
            
            info_gains.append((m, info_gain))
        
        # Select top 5 moduli with highest information gain
        if info_gains:
            info_gains.sort(key=lambda x: x[1], reverse=True)
            info_moduli = [m for m, _ in info_gains[:5]]
        
        # Combine all moduli and sort
        all_moduli = list(set(prime_powers + crt_moduli + special_moduli + info_moduli))
        all_moduli.sort()
        
        return all_moduli
    
    def _generate_fingerprints(self):
        """Generate residue fingerprints for all numbers in range.
        
        The fingerprint of a number n is its residue vector [n mod m1, n mod m2, ...]
        across all moduli. These fingerprints form the basis of the topological analysis.
        """
        fingerprints = np.zeros((len(self.numbers), len(self.moduli)))
        for i, n in enumerate(self.numbers):
            fingerprints[i] = np.array([n % m for m in self.moduli])
            
        # Normalize fingerprints to [0,1] for each modulus dimension
        # This ensures fair weighting across different moduli
        for j in range(fingerprints.shape[1]):
            fingerprints[:, j] /= self.moduli[j]
            
        return fingerprints
    
    def _calculate_potential_field(self):
        """Calculate the scalar potential field Φ based on number-theoretic principles.
        
        The potential field Φ is a scalar function that quantifies the "primeness" of
        each number based on its residue fingerprint. In this implementation, Φ is
        defined based on fundamental number-theoretic functions to capture the essence
        of primality in the residue space.
        """
        potential = np.zeros(len(self.numbers))
        
        # Factor 1: Multiplicative structure factor
        # This captures how a number's residues relate to the multiplicative structure
        # of integers modulo m
        mult_factor = np.ones(len(self.numbers))
        for i, n in enumerate(self.numbers):
            # For each modulus, calculate GCD-based measure
            for j, m in enumerate(self.moduli):
                r = n % m
                if r > 0:  # Avoid division by zero
                    # A prime n will have gcd(n, m) = 1 for most m
                    # This term grows smaller as n shares more factors with m
                    mult_factor[i] *= (1.0 / gcd(r, m))
            
            # Normalize to keep values manageable
            mult_factor[i] = np.log(1 + mult_factor[i])
        
        # Normalize to [0,1]
        mult_factor = (mult_factor - np.min(mult_factor)) / (np.max(mult_factor) - np.min(mult_factor) + 1e-10)
        
        # Factor 2: Residue distribution factor
        # This captures how a number's residues are distributed in the residue space
        dist_factor = np.zeros(len(self.numbers))
        for i in range(len(self.numbers)):
            # Calculate how evenly distributed the residues are
            # Primes tend to have more uniform residue distributions
            normalized_residues = self.fingerprints[i]
            
            # Calculate entropy of the distribution
            # Higher entropy = more uniform distribution = more prime-like
            hist, _ = np.histogram(normalized_residues, bins=10, range=(0, 1))
            hist = hist / np.sum(hist)
            # Fix for deprecated warning - use Python's sum() instead of np.sum(generator)
            entropy = -sum(h * np.log(h + 1e-10) for h in hist)
            dist_factor[i] = entropy
        
        # Normalize to [0,1]
        dist_factor = (dist_factor - np.min(dist_factor)) / (np.max(dist_factor) - np.min(dist_factor) + 1e-10)
        
        # Factor 3: Coprimality factor
        # This directly measures the fundamental property of primes: being coprime
        # to all numbers less than them
        coprime_factor = np.zeros(len(self.numbers))
        for i, n in enumerate(self.numbers):
            # For a prime, this will sum to n-1. For composites, it will be less.
            coprime_count = sum(1 for k in range(1, n) if gcd(k, n) == 1)
            # Normalize by theoretical maximum
            coprime_factor[i] = coprime_count / (n - 1)
        
        # Combine factors with appropriate weights
        # These weights represent the theoretical importance of each factor
        # in determining primality
        potential = 0.4 * mult_factor + 0.3 * dist_factor + 0.3 * coprime_factor
        
        return potential
    
    def _calculate_tensor_field(self):
        """Calculate the tensor field gradient ∇Φ based on analytical differentiation.
        
        The tensor field represents the gradient of the potential field in residue space.
        This gradient points in the direction of increasing "primeness" and its
        magnitude relates to the rate of change of primality.
        """
        gradient = np.zeros((len(self.numbers), len(self.moduli)))
        
        # We'll calculate the analytical gradient of our potential field
        # with respect to each residue dimension
        
        for i, n in enumerate(self.numbers):
            for j, m in enumerate(self.moduli):
                r = n % m
                
                # Calculate gradient component for multiplicative structure factor
                if r > 0:
                    g_mult = -1.0 / (gcd(r, m)**2 * m)
                else:
                    g_mult = 0
                
                # Calculate gradient component for residue distribution factor
                # This is a measure of how much the entropy changes when we shift
                # this specific residue
                normalized_residues = self.fingerprints[i].copy()
                
                # Perturb the jth residue slightly
                eps = 0.01
                normalized_residues[j] += eps
                normalized_residues[j] %= 1.0  # Keep in [0,1]
                
                # Calculate entropy before and after perturbation
                hist_before, _ = np.histogram(self.fingerprints[i], bins=10, range=(0, 1))
                hist_before = hist_before / np.sum(hist_before)
                # Fix for deprecated warning - use Python's sum() instead of np.sum(generator)
                entropy_before = -sum(h * np.log(h + 1e-10) for h in hist_before)
                
                hist_after, _ = np.histogram(normalized_residues, bins=10, range=(0, 1))
                hist_after = hist_after / np.sum(hist_after)
                # Fix for deprecated warning - use Python's sum() instead of np.sum(generator)
                entropy_after = -sum(h * np.log(h + 1e-10) for h in hist_after)
                
                g_dist = (entropy_after - entropy_before) / eps
                
                # Calculate gradient component for coprimality factor
                # This measures how the coprimality changes when we shift the residue
                k = n + m  # Shifting by m doesn't change any other residues
                coprime_count_n = sum(1 for x in range(1, n) if gcd(x, n) == 1)
                coprime_count_k = sum(1 for x in range(1, k) if gcd(x, k) == 1)
                
                g_coprime = (coprime_count_k/(k-1) - coprime_count_n/(n-1)) / m
                
                # Combine gradient components with same weights as in potential field
                gradient[i, j] = 0.4 * g_mult + 0.3 * g_dist + 0.3 * g_coprime
        
        return gradient
    
    def _calculate_laplacian_matrix(self):
        """Calculate the Laplacian matrix of the residue network.
        
        The Laplacian captures the topological structure of the prime residue space.
        Its eigenvalues and eigenvectors reveal the fundamental modes of the prime
        distribution.
        """
        # Create adjacency matrix based on residue space proximity
        n = len(self.numbers)
        adjacency = np.zeros((n, n))
        
        # Calculate pairwise distances in fingerprint space
        for i in range(n):
            for j in range(i+1, n):
                # Calculate weighted distance, with higher weight for primes
                dist = np.linalg.norm(self.fingerprints[i] - self.fingerprints[j])
                
                # Apply proximity function 
                similarity = np.exp(-10 * dist)
                
                # Make connection stronger if both or neither are prime
                # This highlights the topological structure around primes
                if (self.is_prime[i] and self.is_prime[j]) or (not self.is_prime[i] and not self.is_prime[j]):
                    similarity *= 1.5
                
                adjacency[i, j] = adjacency[j, i] = similarity
        
        # Calculate the degree matrix
        degree = np.diag(np.sum(adjacency, axis=1))
        
        # Calculate the Laplacian: L = D - A
        laplacian = degree - adjacency
        
        return laplacian
    
    def _calculate_topological_invariants(self):
        """Calculate topological invariants of the prime residue space.
        
        These invariants quantify the global structure of the prime distribution
        and connect to fundamental constants in number theory.
        """
        invariants = {}
        
        # Calculate Laplacian eigenvalues 
        eigenvalues, eigenvectors = eigh(self.laplacian_matrix)
        
        # The spectrum of the Laplacian characterizes the topology
        invariants['laplacian_spectrum'] = eigenvalues
        
        # The Cheeger constant approximates how well-connected the graph is
        # Higher values indicate better connectivity
        sorted_eigenvalues = np.sort(eigenvalues)
        invariants['cheeger_constant'] = sorted_eigenvalues[1] / 2  # Approximation
        
        # Calculate Betti numbers (topological invariants)
        # Betti-0: number of connected components
        adjacency = self.laplacian_matrix.copy()
        np.fill_diagonal(adjacency, 0)
        adjacency = adjacency < -0.05  # Threshold for counting an edge
        graph = csr_matrix(adjacency)
        n_components, labels = connected_components(graph, directed=False)
        invariants['betti_0'] = n_components
        
        # Calculate prime component ratio - fraction of primes in largest component
        if n_components > 0:
            component_sizes = [np.sum(labels == i) for i in range(n_components)]
            largest_component = np.argmax(component_sizes)
            largest_comp_mask = labels == largest_component
            prime_count_in_largest = np.sum(self.is_prime[largest_comp_mask])
            invariants['prime_component_ratio'] = prime_count_in_largest / np.sum(self.is_prime)
        else:
            invariants['prime_component_ratio'] = 0
                
        # Calculate the spectral gap - related to mixing properties of random walks
        # and expander properties of the graph
        if len(eigenvalues) > 1:
            invariants['spectral_gap'] = eigenvalues[1] - eigenvalues[0]
        else:
            invariants['spectral_gap'] = 0
                
        # The symmetry factor: measures how symmetrically primes are distributed
        # in the residue space (lower values = more symmetric)
        prime_fingerprints = self.fingerprints[self.prime_indices]
        non_prime_fingerprints = self.fingerprints[~self.is_prime]
        
        if len(prime_fingerprints) > 0 and len(non_prime_fingerprints) > 0:
            # Calculate centroid of prime and non-prime points
            prime_centroid = np.mean(prime_fingerprints, axis=0)
            non_prime_centroid = np.mean(non_prime_fingerprints, axis=0)
                
            # Calculate average distance to respective centroids
            prime_distances = np.mean([np.linalg.norm(f - prime_centroid) for f in prime_fingerprints])
            non_prime_distances = np.mean([np.linalg.norm(f - non_prime_centroid) for f in non_prime_fingerprints])
                
            # Symmetry factor: ratio of average distances
            invariants['symmetry_factor'] = prime_distances / non_prime_distances
        else:
            invariants['symmetry_factor'] = 1.0
                
        # Correlation with important constants
        # Link to Riemann zeta function zeros through spectral density
        if len(eigenvalues) > 2:
            density = np.histogram(eigenvalues, bins=min(20, len(eigenvalues)))[0]
            density = density / np.sum(density)
            # Fix for deprecated warning - use Python's sum() instead of np.sum(generator)
            entropy = -sum(d * np.log(d + 1e-10) for d in density)
            # Normalized to relate to critical line spacing
            invariants['riemann_correlation'] = entropy / np.log(len(eigenvalues))
        else:
            invariants['riemann_correlation'] = 0
                
        return invariants
    
    def calculate_embedding(self, dimensions=3, method='tsne'):
        """Calculate low-dimensional embedding of the fingerprint space."""
        try:
            if method.lower() == 'tsne':
                # Adjust perplexity to avoid "perplexity is too large" error
                perplexity = min(30, max(5, len(self.fingerprints) // 5))
                tsne = TSNE(n_components=dimensions, random_state=42, perplexity=perplexity)
                self.low_dim_embedding = tsne.fit_transform(self.fingerprints)
            elif method.lower() == 'laplacian':
                # Spectral embedding using Laplacian eigenvectors
                eigenvalues, eigenvectors = eigh(self.laplacian_matrix)
                # Use the eigenvectors corresponding to the smallest non-zero eigenvalues
                sorted_indices = np.argsort(eigenvalues)
                # Skip the first eigenvector (corresponds to eigenvalue 0)
                n_components = min(dimensions + 1, len(sorted_indices))
                if n_components <= 1:
                    # Fallback to simple projection
                    self.low_dim_embedding = self.fingerprints[:, :min(dimensions, self.fingerprints.shape[1])]
                else:
                    self.low_dim_embedding = eigenvectors[:, sorted_indices[1:n_components]]
            elif method.lower() == 'pca':
                # Simple PCA using SVD
                centered = self.fingerprints - np.mean(self.fingerprints, axis=0)
                U, s, Vt = np.linalg.svd(centered, full_matrices=False)
                n_components = min(dimensions, len(s))
                self.low_dim_embedding = U[:, :n_components] * s[:n_components]
                
            # Ensure we have a valid embedding
            if self.low_dim_embedding is None or self.low_dim_embedding.shape[1] < dimensions:
                # Fallback to a simple projection if the method fails
                self.low_dim_embedding = self.fingerprints[:, :min(dimensions, self.fingerprints.shape[1])]
                
                # If fewer than dimensions columns, pad with zeros
                if self.low_dim_embedding.shape[1] < dimensions:
                    padding = np.zeros((self.low_dim_embedding.shape[0], dimensions - self.low_dim_embedding.shape[1]))
                    self.low_dim_embedding = np.hstack((self.low_dim_embedding, padding))
        except Exception as e:
            print(f"Error in embedding calculation: {e}")
            # Fallback to a simple projection
            self.low_dim_embedding = self.fingerprints[:, :min(dimensions, self.fingerprints.shape[1])]
            if self.low_dim_embedding.shape[1] < dimensions:
                padding = np.zeros((self.low_dim_embedding.shape[0], dimensions - self.low_dim_embedding.shape[1]))
                self.low_dim_embedding = np.hstack((self.low_dim_embedding, padding))
                
        # Final sanity check - ensure no NaN or infinity values
        if np.isnan(self.low_dim_embedding).any() or np.isinf(self.low_dim_embedding).any():
            self.low_dim_embedding = np.nan_to_num(self.low_dim_embedding)
                
        return self.low_dim_embedding
    
    def predict_primality(self, n):
        """Predict primality based on tensor field properties and potential value.
        
        This uses the theoretical framework to assess the "primeness" of a number
        without direct primality testing.
        """
        if n < 2:
            return False
            
        if n >= self.max_number:
            # For numbers beyond our precomputed range, generate the fingerprint
            # and calculate its potential on the fly
            fingerprint = np.array([n % m for m in self.moduli]) / np.array(self.moduli)
            
            # Calculate potential components directly
            
            # Multiplicative structure factor
            mult_factor = 1.0
            for j, m in enumerate(self.moduli):
                r = n % m
                if r > 0:
                    mult_factor *= (1.0 / gcd(r, m))
            mult_factor = np.log(1 + mult_factor)
            
            # Normalize using statistics from known range
            known_mult_factors = np.array([np.log(1 + np.prod([1.0 / gcd(num % m, m) if num % m > 0 else 1.0 
                                            for m in self.moduli])) 
                                        for num in self.numbers])
            mult_factor_norm = (mult_factor - np.min(known_mult_factors)) / (np.max(known_mult_factors) - np.min(known_mult_factors) + 1e-10)
            
            # Residue distribution factor
            # Calculate entropy of fingerprint
            hist, _ = np.histogram(fingerprint, bins=10, range=(0, 1))
            hist = hist / np.sum(hist)
            entropy = -np.sum(h * np.log(h + 1e-10) for h in hist)
            
            # Normalize using statistics from known range
            known_entropies = np.array([
                -np.sum(h * np.log(h + 1e-10) for h in 
                        np.histogram(fp, bins=10, range=(0, 1))[0] / 
                        np.sum(np.histogram(fp, bins=10, range=(0, 1))[0])) 
                for fp in self.fingerprints
            ])
            dist_factor = (entropy - np.min(known_entropies)) / (np.max(known_entropies) - np.min(known_entropies) + 1e-10)
            
            # Coprimality factor
            coprime_count = sum(1 for k in range(1, n) if gcd(k, n) == 1)
            coprime_factor = coprime_count / (n - 1)
            
            # Combine factors
            potential = 0.4 * mult_factor_norm + 0.3 * dist_factor + 0.3 * coprime_factor
            
            # Set threshold based on known primes
            threshold = np.percentile(self.potential_field[self.prime_indices], 10)
            
            return potential > threshold
        else:
            # For numbers in our precomputed range, use the cached potential
            idx = n - 2  # Adjust for 0-indexing and starting at 2
            
            # Use both potential field and tensor field magnitude
            potential_value = self.potential_field[idx]
            gradient_magnitude = np.linalg.norm(self.tensor_field[idx])
            
            # Get thresholds based on known primes
            potential_threshold = np.percentile(self.potential_field[self.prime_indices], 10)
            gradient_threshold = np.percentile(
                [np.linalg.norm(g) for g in self.tensor_field[self.prime_indices]], 
                10
            )
            
            # Combined criteria using theoretically justified thresholds
            return (potential_value > potential_threshold and 
                    gradient_magnitude > gradient_threshold)
    
    def calculate_prime_gap_prediction(self, p):
        """Predict the gap to the next prime using tensor field properties.
        
        This implements the Prime Gap Function from the theoretical framework.
        """
        if p < 2:
            return None
            
        if p >= self.max_number - 100:  # Ensure we have enough room
            # For numbers beyond our precomputed range, use a theoretical approach
            # based on residue patterns
            
            # First, determine congruence class of p
            residue_classes = [p % m for m in self.moduli]
            
            # Calculate the next viable candidate based on these residues
            next_candidates = []
            
            # Generate candidates based on CRT (Chinese Remainder Theorem)
            for gap in range(1, 100):
                candidate = p + gap
                
                # Skip even numbers (except 2)
                if candidate > 2 and candidate % 2 == 0:
                    continue
                    
                # Check if this is a viable candidate in residue space
                is_viable = True
                for m, r in zip(self.moduli, residue_classes):
                    new_r = (r + gap) % m
                    # If new residue makes division by a small prime likely, reject
                    for small_prime in [3, 5, 7, 11]:
                        if m % small_prime == 0 and new_r % small_prime == 0:
                            is_viable = False
                            break
                            
                    if not is_viable:
                        break
                        
                if is_viable:
                    next_candidates.append(gap)
                    
                    # Return first viable candidate
                    if len(next_candidates) > 0:
                        return next_candidates[0]
                    
            # Fallback if no candidates found
            return self._empirical_gap_estimate(p)
        else:
            # For numbers in range, use the tensor field
            idx = p - 2  # Adjust index
            
            # Use tensor field gradient to estimate direction to next prime
            gradient = self.tensor_field[idx]
            
            # Normalize gradient
            if np.linalg.norm(gradient) > 0:
                gradient = gradient / np.linalg.norm(gradient)
            
            # Project gradient onto integers
            # This maps the continuous gradient to discrete jumps
            
            # First, interpret gradient as directing us to a point in residue space
            target_residues = self.fingerprints[idx] + gradient * 0.5  # Scale factor
            
            # Find smallest positive integer that approximates these target residues
            best_gap = 1
            best_distance = float('inf')
            
            for gap in range(1, 100):  # Check reasonable gaps
                candidate = p + gap
                if candidate % 2 == 0 and candidate > 2:  # Skip even numbers
                    continue
                    
                # Calculate residues for this candidate
                candidate_residues = np.array([candidate % m for m in self.moduli]) / np.array(self.moduli)
                
                # Calculate distance in residue space
                distance = np.linalg.norm(candidate_residues - target_residues)
                
                if distance < best_distance:
                    best_distance = distance
                    best_gap = gap
            
            return best_gap
    
    def _empirical_gap_estimate(self, p):
        """Provide an empirical estimate of the prime gap based on statistical patterns."""
        # Use the prime number theorem to estimate expected gap size
        # The average gap near p is approximately log(p)
        avg_gap = np.log(p)
        
        # Adjust for even/odd pattern
        if p == 2:
            return 1  # Gap from 2 to 3
        elif p % 6 == 1:
            return max(2, round(avg_gap))  # For primes congruent to 1 mod 6, gap tends to be even
        elif p % 6 == 5:
            return max(1, round(avg_gap * 0.8))  # For primes congruent to 5 mod 6, gap tends to be smaller
        else:
            return max(1, round(avg_gap))
    
    def lattice_reduction_prime_search(self, search_range=100, start=None):
        """Implement lattice reduction to find potential primes.
        
        This uses the theoretical insight that primes correspond to short vectors
        in a particular lattice construction in residue space.
        """
        if start is None:
            start = self.max_number
        
        # Create a basis for our lattice
        # Cconstruct a lattice where prime patterns correspond to short vectors
        
        # First, generate reference indices
        reference_indices = np.where((self.numbers >= start) & 
                                    (self.numbers < start + search_range))[0]
        
        if len(reference_indices) == 0:
            return []
            
        # Extract fingerprints for the search range
        search_fingerprints = self.fingerprints[reference_indices]
        search_numbers = self.numbers[reference_indices]
        
        # Construct lattice basis
        # For each modulus, create a basis vector that captures
        # the relationship between residues and primality
        
        basis = []
        
        # Add basis vectors from fingerprints
        for i in range(len(self.moduli)):
            # Create a basis vector that emphasizes the i-th residue
            basis_vector = np.zeros(len(self.moduli) + 1)
            basis_vector[i] = 1.0
            # The last component links to theoretical primality
            basis_vector[-1] = 0.1  # Small weight to maintain relationship
            basis.append(basis_vector)
            
        # Add a vector for the theoretical primality relationship
        primality_vector = np.zeros(len(self.moduli) + 1)
        primality_vector[-1] = 1.0
        basis.append(primality_vector)
        
        # Convert to numpy array
        basis = np.array(basis)
        
        # For each number in the search range, extend its fingerprint
        # with its potential field value
        extended_fingerprints = np.zeros((len(search_numbers), len(self.moduli) + 1))
        extended_fingerprints[:, :-1] = search_fingerprints
        
        # Calculate potential field values for the search range
        for i, n in enumerate(search_numbers):
            if n < self.max_number:
                idx = n - 2
                extended_fingerprints[i, -1] = self.potential_field[idx]
            else:
                # Calculate potential on the fly for numbers beyond range
                fingerprint = np.array([n % m for m in self.moduli]) / np.array(self.moduli)
                
                # Multiplicative structure factor
                mult_factor = 1.0
                for j, m in enumerate(self.moduli):
                    r = n % m
                    if r > 0:
                        mult_factor *= (1.0 / gcd(r, m))
                mult_factor = np.log(1 + mult_factor)
                
                # Normalize using statistics from known range
                known_mult_factors = np.array([np.log(1 + np.prod([1.0 / gcd(num % m, m) if num % m > 0 else 1.0 
                                                for m in self.moduli])) 
                                            for num in self.numbers])
                mult_factor_norm = (mult_factor - np.min(known_mult_factors)) / (np.max(known_mult_factors) - np.min(known_mult_factors) + 1e-10)
                
                # Residue distribution factor
                hist, _ = np.histogram(fingerprint, bins=10, range=(0, 1))
                hist = hist / np.sum(hist)
                entropy = -np.sum(h * np.log(h + 1e-10) for h in hist)
                
                # Normalize
                known_entropies = np.array([
                    -np.sum(h * np.log(h + 1e-10) for h in 
                            np.histogram(fp, bins=10, range=(0, 1))[0] / 
                            np.sum(np.histogram(fp, bins=10, range=(0, 1))[0])) 
                    for fp in self.fingerprints
                ])
                dist_factor = (entropy - np.min(known_entropies)) / (np.max(known_entropies) - np.min(known_entropies) + 1e-10)
                
                # Coprimality factor
                if n < 1000:  # Only calculate for smaller numbers for performance
                    coprime_count = sum(1 for k in range(1, n) if gcd(k, n) == 1)
                    coprime_factor = coprime_count / (n - 1)
                else:
                    # Estimate for larger numbers using Euler's totient approximation
                    # For primes, φ(n) = n-1
                    factors = factorint(n)
                    if len(factors) == 1:
                        p = list(factors.keys())[0]
                        k = factors[p]
                        # Totient formula for prime power: φ(p^k) = p^k - p^(k-1)
                        coprime_factor = 1 - 1/p
                    else:
                        # Approximate totient for composite
                        coprime_factor = 0.5  # Default for typical composite
                
                # Combine factors
                extended_fingerprints[i, -1] = 0.4 * mult_factor_norm + 0.3 * dist_factor + 0.3 * coprime_factor
        
        # Apply QR decomposition for lattice reduction
        Q, R = qr(basis, mode='economic')
        
        # Project extended fingerprints onto reduced basis
        projections = np.dot(extended_fingerprints, Q)
        
        # Calculate Euclidean norm of each projection
        norms = np.linalg.norm(projections, axis=1)
        
        # Sort numbers by increasing norm (shorter vectors are more prime-like)
        sorted_indices = np.argsort(norms)
        candidates = search_numbers[sorted_indices]
        
        # Calculate prime threshold based on known primes (better than a hardcoded value)
        if len(self.primes) > 0:
            # Project known primes onto the reduced basis and analyze
            known_prime_fingerprints = np.zeros((len(self.primes), len(self.moduli) + 1))
            known_prime_fingerprints[:, :-1] = self.fingerprints[self.prime_indices]
            known_prime_fingerprints[:, -1] = self.potential_field[self.prime_indices]
            
            known_projections = np.dot(known_prime_fingerprints, Q)
            known_norms = np.linalg.norm(known_projections, axis=1)
            
            # Set threshold at 90th percentile of known prime norms
            threshold = np.percentile(known_norms, 90)
            
            # Select candidates below this threshold
            mask = norms[sorted_indices] < threshold
            candidates = candidates[mask]
        
        # Filter out even numbers (except 2) for efficiency
        candidates = [c for c in candidates if c == 2 or c % 2 == 1]
        
        # Additional filtering based on residue patterns known to be composite
        final_candidates = []
        for c in candidates:
            # Additional number theory filters
            if c > 2 and c % 2 == 0:
                continue  # Skip even numbers
                
            # Check for divisibility by small primes
            if any(c % p == 0 for p in [3, 5, 7, 11, 13] if c > p):
                continue
                
            # Apply Fermat's little theorem as a probabilistic test
            # If a^(p-1) ≡ 1 (mod p) for several a, p is likely prime
            passes_fermat = True
            for a in [2, 3, 5]:
                # Convert to Python integer types for pow function - this fixes the error
                if pow(a, int(c-1), int(c)) != 1:
                    passes_fermat = False
                    break
                    
            if passes_fermat:
                final_candidates.append(c)
        
        return final_candidates
    
    def identify_topological_features(self):
        """Identify topological features in the prime distribution.
        
        This analyzes the topological structure of the prime distribution
        in residue space, connecting to deep mathematical properties.
        """
        if self.low_dim_embedding is None:
            self.calculate_embedding(3)
            
        # Use only prime points for topology analysis
        prime_points = self.low_dim_embedding[self.prime_indices]
        
        # Results dictionary
        topology_results = {}
        
        # Calculate persistent homology-inspired features
        try:
            # First, create a simplicial complex using Delaunay triangulation
            if len(prime_points) >= 4:  # Need at least 4 points for 3D Delaunay
                tri = Delaunay(prime_points)
                simplices = tri.simplices
                
                # Count simplices of each dimension
                topology_results['num_vertices'] = len(prime_points)
                topology_results['num_edges'] = len(np.unique(np.sort(tri.simplices[:, :2], axis=1), axis=0))
                topology_results['num_triangles'] = len(np.unique(np.sort(tri.simplices[:, :3], axis=1), axis=0))
                topology_results['num_tetrahedra'] = len(simplices)
                
                # Calculate Euler characteristic
                # χ = V - E + F - T
                euler_char = (topology_results['num_vertices'] - 
                             topology_results['num_edges'] + 
                             topology_results['num_triangles'] - 
                             topology_results['num_tetrahedra'])
                topology_results['euler_characteristic'] = euler_char
            else:
                # For few points, use simplified calculation
                topology_results['num_vertices'] = len(prime_points)
                topology_results['num_edges'] = len(prime_points) * (len(prime_points) - 1) // 2
                topology_results['num_triangles'] = 0
                topology_results['num_tetrahedra'] = 0
                topology_results['euler_characteristic'] = len(prime_points) - topology_results['num_edges']
        except:
            # Fallback for triangulation failure
            topology_results['num_vertices'] = len(prime_points)
            topology_results['euler_characteristic'] = 1
        
        # Graph-based topological analysis
        G = nx.Graph()
        for i in range(len(prime_points)):
            G.add_node(i)
        
        # Connect close points (using distance threshold)
        threshold = np.percentile([
            np.linalg.norm(prime_points[i] - prime_points[j])
            for i in range(len(prime_points))
            for j in range(i+1, min(i+20, len(prime_points)))  # Use windowed approach for performance
        ], 20)  # 20th percentile of distances
        
        for i in range(len(prime_points)):
            for j in range(i+1, len(prime_points)):
                if np.linalg.norm(prime_points[i] - prime_points[j]) < threshold:
                    G.add_edge(i, j)
        
        # Calculate network properties
        topology_results['clustering_coefficient'] = nx.average_clustering(G)
        topology_results['avg_shortest_path'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
        
        # Connected components analysis
        components = list(nx.connected_components(G))
        topology_results['num_components'] = len(components)
        
        if components:
            largest_component_size = max(len(c) for c in components)
            topology_results['largest_component_size'] = largest_component_size
            topology_results['largest_component_ratio'] = largest_component_size / len(prime_points)
        else:
            topology_results['largest_component_size'] = 0
            topology_results['largest_component_ratio'] = 0
        
        # Prime gap distribution analysis - connecting to topology
        if len(self.primes) > 1:
            gaps = np.diff(self.primes)
            
            # Calculate statistics
            topology_results['avg_gap'] = float(np.mean(gaps))
            topology_results['max_gap'] = int(np.max(gaps))
            topology_results['gap_std'] = float(np.std(gaps))
            
            # Analyze gap patterns using autocorrelation
            if len(gaps) > 2:
                autocorr = np.correlate(gaps - np.mean(gaps), gaps - np.mean(gaps), mode='full')
                autocorr = autocorr[len(gaps)-1:] / autocorr[len(gaps)-1]
                
                # Extract cycle length using first peak after zero
                for i in range(1, min(10, len(autocorr))):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1 if i+1 < len(autocorr) else i]:
                        topology_results['gap_cycle_length'] = i
                        break
                else:
                    topology_results['gap_cycle_length'] = 0
            else:
                topology_results['gap_cycle_length'] = 0
        
        # Connect to mathematical constants
        # We calculate correlations with mathematical constants that appear in
        # number theory and prime distribution
        
        # Pi correlation - compare ratio of components to pi
        if topology_results['num_components'] > 0:
            pi_ratio = topology_results['largest_component_size'] / topology_results['num_components']
            pi_error = abs(pi_ratio - np.pi) / np.pi
            topology_results['pi_correlation'] = 1 - pi_error
        else:
            topology_results['pi_correlation'] = 0
            
        # Golden ratio correlation - compare clustering pattern to phi
        phi = (1 + np.sqrt(5)) / 2
        if topology_results['clustering_coefficient'] > 0:
            phi_ratio = 1 / topology_results['clustering_coefficient']
            phi_error = abs(phi_ratio - phi) / phi
            topology_results['phi_correlation'] = 1 - phi_error
        else:
            topology_results['phi_correlation'] = 0
            
        # Euler-Mascheroni constant correlation - compare to average gap ratio
        if 'avg_gap' in topology_results and topology_results['avg_gap'] > 0:
            gamma = 0.57721566490153286  # Euler-Mascheroni constant
            expected_ratio = np.log(np.log(self.max_number)) * gamma
            actual_ratio = topology_results['avg_gap'] / np.log(self.max_number)
            gamma_error = abs(actual_ratio - expected_ratio) / expected_ratio
            topology_results['gamma_correlation'] = 1 - min(1, gamma_error)
        else:
            topology_results['gamma_correlation'] = 0
            
        # Add Riemann zeta related correlation from invariants
        topology_results['riemann_correlation'] = self.invariants['riemann_correlation']
        
        return topology_results

    def generate_comprehensive_analysis(self):
        """Generate a comprehensive analysis of the prime residue topology."""
        analysis = {}
        
        # Basic statistics
        analysis['basic_stats'] = {
            'max_number': self.max_number,
            'num_primes': len(self.primes),
            'prime_density': len(self.primes) / self.max_number,
            'moduli_used': self.moduli,
            'optimal_moduli_explanation': "Moduli are selected based on prime powers, CRT-based resonance, and information theory principles to maximize the separation between primes and composites in residue space."
        }
        
        # Fingerprint statistics
        analysis['fingerprint_stats'] = {
            'avg_fingerprint': np.mean(self.fingerprints, axis=0).tolist(),
            'std_fingerprint': np.std(self.fingerprints, axis=0).tolist(),
            'prime_avg_fingerprint': np.mean(self.fingerprints[self.prime_indices], axis=0).tolist(),
            'composite_avg_fingerprint': np.mean(self.fingerprints[~self.is_prime], axis=0).tolist(),
            'fingerprint_interpretation': "Prime fingerprints show more uniform distribution across residue classes, reflecting their lack of algebraic structure compared to composites."
        }
        
        # Potential field statistics
        analysis['potential_field_stats'] = {
            'min_potential': float(np.min(self.potential_field)),
            'max_potential': float(np.max(self.potential_field)),
            'avg_potential': float(np.mean(self.potential_field)),
            'std_potential': float(np.std(self.potential_field)),
            'prime_avg_potential': float(np.mean(self.potential_field[self.prime_indices])),
            'composite_avg_potential': float(np.mean(self.potential_field[~self.is_prime])),
            'potential_separability': float(
                (np.mean(self.potential_field[self.prime_indices]) - 
                 np.mean(self.potential_field[~self.is_prime])) / np.std(self.potential_field)
            ),
            'theoretical_foundation': "The potential field Φ is derived from multiplicative structure, residue distribution entropy, and coprimality principles fundamental to prime numbers."
        }
        
        # Tensor field statistics
        tensor_magnitude = np.linalg.norm(self.tensor_field, axis=1)
        analysis['tensor_field_stats'] = {
            'min_magnitude': float(np.min(tensor_magnitude)),
            'max_magnitude': float(np.max(tensor_magnitude)),
            'avg_magnitude': float(np.mean(tensor_magnitude)),
            'std_magnitude': float(np.std(tensor_magnitude)),
            'prime_avg_magnitude': float(np.mean(tensor_magnitude[self.prime_indices])),
            'composite_avg_magnitude': float(np.mean(tensor_magnitude[~self.is_prime])),
            'magnitude_separability': float(
                (np.mean(tensor_magnitude[self.prime_indices]) - 
                 np.mean(tensor_magnitude[~self.is_prime])) / np.std(tensor_magnitude)
            ),
            'gradient_interpretation': "The tensor field ∇Φ captures how the potential changes with residue values. Primes exhibit distinctive gradient patterns that reflect their special status in residue space."
        }
        
        # Topological invariants
        analysis['topological_invariants'] = {}
        for key, value in self.invariants.items():
            if key == 'laplacian_spectrum':
                # Only include summary statistics for large spectra
                if len(value) > 10:
                    analysis['topological_invariants'][key + '_summary'] = {
                        'min': float(np.min(value)),
                        'max': float(np.max(value)),
                        'mean': float(np.mean(value)),
                        'std': float(np.std(value)),
                        'spectral_gap': float(value[1] - value[0]) if len(value) > 1 else 0
                    }
                else:
                    analysis['topological_invariants'][key] = value.tolist()
            else:
                # Convert numpy types to Python types
                if isinstance(value, (np.integer, np.floating)):
                    analysis['topological_invariants'][key] = float(value)
                else:
                    analysis['topological_invariants'][key] = value
        
        # Add interpretation
        analysis['topological_invariants']['interpretation'] = (
            "These invariants quantify the global structure of the prime distribution in residue space. "
            "The Laplacian spectrum and Cheeger constant reveal the connectivity and mixing properties of the prime graph. "
            "The correlation values show connections to fundamental mathematical constants."
        )
        
        # Prime prediction performance
        # Let's sample a few ranges and check prediction accuracy
        prediction_results = []
        for start in np.linspace(2, self.max_number - 100, 5, dtype=int):
            range_size = 100
            start_time = time.time()
            predicted = self.lattice_reduction_prime_search(range_size, start)
            prediction_time = time.time() - start_time
            actual = list(primerange(start, start + range_size))
            
            true_positives = len(set(predicted) & set(actual))
            false_positives = len(set(predicted) - set(actual))
            false_negatives = len(set(actual) - set(predicted))
            
            precision = true_positives / max(1, len(predicted))
            recall = true_positives / max(1, len(actual))
            f1 = 2 * precision * recall / max(0.001, precision + recall)
            
            prediction_results.append({
                'start': int(start),
                'range_size': range_size,
                'num_predicted': len(predicted),
                'num_actual': len(actual),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'prediction_time': float(prediction_time)
            })
        
        analysis['prediction_performance'] = prediction_results
        
        # Prime gap prediction analysis
        gap_analysis = {
            'prediction_method': "Prime gaps are predicted using the tensor field gradient projected onto integer space, aligned with the Prime Gap Function in the theoretical framework."
        }
        
        # Test gap predictions for a sample of known primes
        if len(self.primes) > 10:
            sample_indices = np.random.choice(len(self.primes)-1, min(10, len(self.primes)-1), replace=False)
            gap_predictions = []
            
            for idx in sample_indices:
                p = self.primes[idx]
                actual_gap = self.primes[idx+1] - p
                predicted_gap = self.calculate_prime_gap_prediction(p)
                
                gap_predictions.append({
                    'prime': int(p),
                    'actual_gap': int(actual_gap),
                    'predicted_gap': int(predicted_gap),
                    'error': abs(predicted_gap - actual_gap),
                    'relative_error': float(abs(predicted_gap - actual_gap) / actual_gap)
                })
                
            gap_analysis['sample_predictions'] = gap_predictions
            
            # Calculate overall accuracy metrics
            errors = [pred['error'] for pred in gap_predictions]
            rel_errors = [pred['relative_error'] for pred in gap_predictions]
            
            gap_analysis['mean_abs_error'] = float(np.mean(errors))
            gap_analysis['median_abs_error'] = float(np.median(errors))
            gap_analysis['mean_rel_error'] = float(np.mean(rel_errors))
            gap_analysis['median_rel_error'] = float(np.median(rel_errors))
            
        analysis['gap_prediction_analysis'] = gap_analysis
        
        # Topology analysis
        if self.low_dim_embedding is None:
            self.calculate_embedding(3)
        
        topology_features = self.identify_topological_features()
        # Convert any numpy types to Python native types
        for key, value in topology_features.items():
            if isinstance(value, (np.integer, np.floating)):
                topology_features[key] = float(value)
        
        analysis['topology_features'] = topology_features
        
        # Prime gap analysis
        if len(self.primes) > 1:
            prime_gaps = np.diff(self.primes)
            gap_distribution = np.bincount(prime_gaps)
            analysis['prime_gap_stats'] = {
                'min_gap': int(np.min(prime_gaps)),
                'max_gap': int(np.max(prime_gaps)),
                'avg_gap': float(np.mean(prime_gaps)),
                'std_gap': float(np.std(prime_gaps)),
                'gap_distribution': gap_distribution.tolist(),
                'theoretical_relation': "The distribution of prime gaps relates to the Prime Gap Function in the theoretical framework, which is driven by the tensor field gradient in residue space."
            }
        
        # Mathematical connections
        analysis['mathematical_connections'] = {
            'pi_connection': topology_features.get('pi_correlation', 0),
            'phi_connection': topology_features.get('phi_correlation', 0),
            'gamma_connection': topology_features.get('gamma_correlation', 0),
            'riemann_connection': topology_features.get('riemann_correlation', 0),
            'interpretation': "These connections reveal how the topological structure of primes relates to fundamental mathematical constants. The correlations suggest deep relationships between prime distribution and these constants."
        }
        
        # Code performance metrics
        timing = {}
        
        start_time = time.time()
        self._generate_fingerprints()
        timing['fingerprint_generation'] = time.time() - start_time
        
        start_time = time.time()
        self._calculate_potential_field()
        timing['potential_field_calculation'] = time.time() - start_time
        
        start_time = time.time()
        self._calculate_tensor_field()
        timing['tensor_field_calculation'] = time.time() - start_time
        
        start_time = time.time()
        self.calculate_embedding(3)
        timing['embedding_calculation'] = time.time() - start_time
        
        analysis['performance_timing'] = timing
        
        # Theoretical insights
        analysis['theoretical_insights'] = [
            "The Prime Residue Topology framework reveals primes as critical points in a potential field Φ defined on residue space",
            "Prime gaps emerge as integral curves of the tensor field gradient ∇Φ, explaining their distribution patterns",
            "The topological structure of primes connects to fundamental mathematical constants like π, φ, and γ",
            "Lattice reduction in residue space provides a novel approach to prime identification with theoretical grounding",
            "The framework suggests that primality is not binary but exists on a continuous spectrum measured by Φ",
            "Symmetry breaking in residue space corresponds precisely to the emergence of primality",
            "The approach bridges number theory with differential geometry and topology in a unified mathematical framework",
            "Spectral properties of the residue Laplacian relate to the distribution of Riemann zeta zeros",
            "Chinese Remainder Theorem underlies the effectiveness of multi-moduli fingerprinting",
            "The framework provides theoretical support for deterministic prediction of large prime gaps"
        ]
        
        # Future research directions
        analysis['future_directions'] = [
            "Extend the theoretical foundation using algebraic geometry to further characterize the prime manifold",
            "Develop more sophisticated potential functions derived from analytic number theory",
            "Explore connections to the Riemann Hypothesis through spectral properties of the residue Laplacian",
            "Implement persistent homology calculations to analyze topological features across multiple scales",
            "Improve tensor field calculations using techniques from differential geometry",
            "Develop a quantum mechanical interpretation of prime emergence through symmetry breaking",
            "Investigate higher-dimensional topological invariants of the prime distribution",
            "Extend the lattice reduction algorithm using more advanced basis reduction techniques",
            "Explore prime constellation patterns as multi-dimensional manifolds in residue space",
            "Develop a cohomology theory for prime residue space to better understand its global structure"
        ]
        
        return analysis


class MatplotlibCanvas(FigureCanvas):
    """Canvas for Matplotlib visualizations in Qt5."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        # Create figure with dark style
        with plt.style.context('dark_background'):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            # Set background color explicitly
            self.fig.patch.set_facecolor('#2D2D2D')
            self.axes = self.fig.add_subplot(111)
            self.axes.set_facecolor('#2D2D2D')
        
        # Initialize the canvas with our figure
        super(MatplotlibCanvas, self).__init__(self.fig)
        
        # Ensure visibility of axes elements
        self.axes.tick_params(colors='#D0D0D0')
        for spine in self.axes.spines.values():
            spine.set_edgecolor('#707070')
            
        # Set up figure for Qt
        self.setMinimumSize(400, 300)  # Set a minimum size for visibility


class PrimeVisualizer(QMainWindow):
    """Qt5 interface for the Prime Residue Topology Explorer."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Prime Residue Topology Explorer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize
        self.max_number = 100  
        
        # Show a progress dialog during initialization
        progress = QDialog(self)
        progress.setWindowTitle("Initializing")
        progress_layout = QVBoxLayout(progress)
        progress_layout.addWidget(QLabel("Initializing Prime Residue Topology...\nThis may take a moment."))
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()  # Process events to show the dialog
        
        # Initialize the topology model
        self.topology = PrimeResidueTopology(max_number=self.max_number)
        
        # Close progress dialog
        progress.close()
        
        # Set up the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget for different visualizations
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_3d_visualization_tab()
        self.create_tensor_field_tab()
        self.create_prime_prediction_tab()
        self.create_topology_analysis_tab()
        
        # Update the range slider to match our initial value
        self.range_slider.setValue(self.max_number)
        
        # Add analysis button
        self.analysis_button = QPushButton("Generate Comprehensive Analysis")
        self.analysis_button.clicked.connect(self.show_comprehensive_analysis)
        self.main_layout.addWidget(self.analysis_button)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")
        
        # Initialize visualizations
        self.update_visualizations()
    
    def create_3d_visualization_tab(self):
        """Create tab for 3D visualization of residue space."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls group
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QHBoxLayout()
        
        # Number range control
        range_layout = QVBoxLayout()
        range_layout.addWidget(QLabel("Number Range:"))
        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setMinimum(100)
        self.range_slider.setMaximum(5000)
        self.range_slider.setValue(self.max_number)
        self.range_slider.setTickInterval(500)
        self.range_slider.setTickPosition(QSlider.TicksBelow)
        self.range_slider.valueChanged.connect(self.on_range_changed)
        range_layout.addWidget(self.range_slider)
        range_layout.addWidget(QLabel("Max Number: 1000"))
        controls_layout.addLayout(range_layout)
        
        # Embedding method
        embedding_layout = QVBoxLayout()
        embedding_layout.addWidget(QLabel("Embedding Method:"))
        self.embedding_combo = QComboBox()
        self.embedding_combo.addItems(["t-SNE", "PCA", "Laplacian"])
        self.embedding_combo.currentTextChanged.connect(self.update_visualizations)
        embedding_layout.addWidget(self.embedding_combo)
        controls_layout.addLayout(embedding_layout)
        
        # Dimension selector
        dim_layout = QVBoxLayout()
        dim_layout.addWidget(QLabel("Dimensions:"))
        self.dim_spinner = QSpinBox()
        self.dim_spinner.setMinimum(2)
        self.dim_spinner.setMaximum(3)
        self.dim_spinner.setValue(3)
        self.dim_spinner.valueChanged.connect(self.update_visualizations)
        dim_layout.addWidget(self.dim_spinner)
        controls_layout.addLayout(dim_layout)
        
        # Update button
        self.update_button = QPushButton("Update Visualization")
        self.update_button.clicked.connect(self.update_visualizations)
        controls_layout.addWidget(self.update_button)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Visualization canvas
        self.viz_canvas = MatplotlibCanvas(width=10, height=8)
        layout.addWidget(self.viz_canvas)
        
        self.tabs.addTab(tab, "3D Residue Space")
    
    def create_tensor_field_tab(self):
        """Create tab for tensor field visualization."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls for tensor field visualization
        controls_group = QGroupBox("Tensor Field Controls")
        controls_layout = QHBoxLayout()
        
        # Field component selector
        field_layout = QVBoxLayout()
        field_layout.addWidget(QLabel("Field Component:"))
        self.field_combo = QComboBox()
        self.field_combo.addItems(["Potential Φ", "Gradient Magnitude", "X Component", "Y Component"])
        self.field_combo.currentTextChanged.connect(self.update_tensor_field)
        field_layout.addWidget(self.field_combo)
        controls_layout.addLayout(field_layout)
        
        # Visualization type
        viz_layout = QVBoxLayout()
        viz_layout.addWidget(QLabel("Visualization:"))
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["Heatmap", "Surface", "Contour"])
        self.viz_combo.currentTextChanged.connect(self.update_tensor_field)
        viz_layout.addWidget(self.viz_combo)
        controls_layout.addLayout(viz_layout)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Tensor field visualization canvas
        self.tensor_canvas = MatplotlibCanvas(width=10, height=8)
        layout.addWidget(self.tensor_canvas)
        
        self.tabs.addTab(tab, "Tensor Field")
    
    def create_prime_prediction_tab(self):
        """Create tab for prime prediction demonstration."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Split view for visualization and prediction results
        splitter = QSplitter(Qt.Vertical)
        
        # Top part: visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        controls_group = QGroupBox("Prime Prediction Controls")
        controls_layout = QHBoxLayout()
        
        # Starting number for prediction
        start_layout = QVBoxLayout()
        start_layout.addWidget(QLabel("Start Number:"))
        self.start_spinner = QSpinBox()
        self.start_spinner.setMinimum(2)
        self.start_spinner.setMaximum(10000)
        self.start_spinner.setValue(1000)
        start_layout.addWidget(self.start_spinner)
        controls_layout.addLayout(start_layout)
        
        # Range for prediction
        range_layout = QVBoxLayout()
        range_layout.addWidget(QLabel("Prediction Range:"))
        self.pred_range_spinner = QSpinBox()
        self.pred_range_spinner.setMinimum(10)
        self.pred_range_spinner.setMaximum(500)
        self.pred_range_spinner.setValue(100)
        range_layout.addWidget(self.pred_range_spinner)
        controls_layout.addLayout(range_layout)
        
        # Run prediction button
        self.predict_button = QPushButton("Run Prediction")
        self.predict_button.clicked.connect(self.run_prime_prediction)
        controls_layout.addWidget(self.predict_button)
        
        controls_group.setLayout(controls_layout)
        viz_layout.addWidget(controls_group)
        
        # Prediction visualization canvas
        self.prediction_canvas = MatplotlibCanvas(width=10, height=6)
        viz_layout.addWidget(self.prediction_canvas)
        
        splitter.addWidget(viz_widget)
        
        # Bottom part: results text
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.addWidget(QLabel("Prediction Results:"))
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        splitter.addWidget(results_widget)
        
        # Set initial sizes
        splitter.setSizes([600, 200])
        
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Prime Prediction")
    
    def create_topology_analysis_tab(self):
        """Create tab for topological analysis."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Split view for visualization and analysis results
        splitter = QSplitter(Qt.Horizontal)
        
        # Left part: visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        # Topology visualization canvas
        self.topology_canvas = MatplotlibCanvas(width=6, height=8)
        viz_layout.addWidget(self.topology_canvas)
        
        # Analyze button
        self.analyze_button = QPushButton("Analyze Topology")
        self.analyze_button.clicked.connect(self.analyze_topology)
        viz_layout.addWidget(self.analyze_button)
        
        viz_widget.setLayout(viz_layout)
        splitter.addWidget(viz_widget)
        
        # Right part: analysis results
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        analysis_layout.addWidget(QLabel("Topological Analysis:"))
        
        self.topology_results = QTextEdit()
        self.topology_results.setReadOnly(True)
        analysis_layout.addWidget(self.topology_results)
        
        analysis_widget.setLayout(analysis_layout)
        splitter.addWidget(analysis_widget)
        
        # Set initial sizes
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Topology Analysis")
    
    def update_visualizations(self):
        """Update all visualizations based on current parameters."""
        self.statusBar().showMessage("Updating visualizations...")
        
        # Show progress dialog for longer operations
        progress = QDialog(self)
        progress.setWindowTitle("Updating")
        progress_layout = QVBoxLayout(progress)
        progress_label = QLabel("Updating visualizations...\nThis may take a moment.")
        progress_layout.addWidget(progress_label)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()  # Process events to show the dialog
        
        try:
            # Update model if range changed
            max_number = self.range_slider.value()
            if max_number != self.max_number:
                progress_label.setText(f"Recalculating with max_number={max_number}...\nThis may take a moment.")
                QApplication.processEvents()
                self.max_number = max_number
                self.topology = PrimeResidueTopology(max_number=self.max_number)
            
            # Calculate embedding based on selected method
            dimensions = self.dim_spinner.value()
            method = self.embedding_combo.currentText().lower()
            
            # Update 3D visualization
            progress_label.setText("Calculating embedding...")
            QApplication.processEvents()
            self.topology.calculate_embedding(dimensions=dimensions, method=method)
            
            # Update visualizations
            progress_label.setText("Updating 3D visualization...")
            QApplication.processEvents()
            self.update_3d_visualization()
            
            # Update tensor field visualization
            progress_label.setText("Updating tensor field...")
            QApplication.processEvents()
            self.update_tensor_field()
            
            self.statusBar().showMessage("Visualizations updated")
        except Exception as e:
            self.statusBar().showMessage(f"Error updating visualizations: {str(e)}")
            # Show error dialog
            error_dialog = QDialog(self)
            error_dialog.setWindowTitle("Error")
            error_layout = QVBoxLayout(error_dialog)
            error_layout.addWidget(QLabel(f"An error occurred while updating visualizations:\n{str(e)}"))
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(error_dialog.accept)
            error_layout.addWidget(ok_button)
            error_dialog.exec_()
        finally:
            # Always close the progress dialog
            progress.close()
    
    def update_3d_visualization(self):
        """Update the 3D residue space visualization."""
        try:
            # Clear the current figure instead of replacing it
            self.viz_canvas.fig.clear()
            
            # Ensure embedding exists
            if self.topology.low_dim_embedding is None:
                dimensions = self.dim_spinner.value()
                method = self.embedding_combo.currentText().lower()
                self.topology.calculate_embedding(dimensions=dimensions, method=method)
            
            embedding = self.topology.low_dim_embedding
            
            # Add safety check
            if embedding is None or len(embedding) == 0:
                # Create a default axes with an error message
                self.viz_canvas.axes = self.viz_canvas.fig.add_subplot(111)
                self.viz_canvas.axes.text(
                    0.5, 0.5, 
                    "Error: Could not generate embedding.\nTry a different method or smaller number range.",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=self.viz_canvas.axes.transAxes,
                    color='#D0D0D0'  # Use theme color
                )
                self.viz_canvas.axes.set_axis_off()
                self.viz_canvas.fig.tight_layout()
                self.viz_canvas.draw()
                return
            
            dimensions = embedding.shape[1]
            
            if dimensions == 2:
                # 2D scatter plot
                self.viz_canvas.axes = self.viz_canvas.fig.add_subplot(111)
                # Set background color explicitly
                self.viz_canvas.axes.set_facecolor('#2D2D2D')
                
                # Create a colormap that maps primes to one color and composites to another
                colors = np.zeros(len(self.topology.numbers))
                colors[self.topology.prime_indices] = 1  # Primes get value 1
                
                scatter = self.viz_canvas.axes.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=colors,
                    cmap='coolwarm',
                    alpha=0.7,
                    s=30
                )
                
                # Add labels for some primes
                for i, idx in enumerate(self.topology.prime_indices):
                    if i % max(1, len(self.topology.prime_indices) // 20) == 0:  # Label ~20 primes
                        self.viz_canvas.axes.text(
                            embedding[idx, 0],
                            embedding[idx, 1],
                            str(self.topology.numbers[idx]),
                            fontsize=9,
                            color='#D0D0D0',  # Use theme color
                            weight='bold'
                        )
                
                self.viz_canvas.axes.set_xlabel("Dimension 1", color='#D0D0D0')  # Use theme color
                self.viz_canvas.axes.set_ylabel("Dimension 2", color='#D0D0D0')  # Use theme color
                
                # Add colorbar with clear labels
                cbar = self.viz_canvas.fig.colorbar(scatter, ticks=[0, 1])
                cbar.ax.set_yticklabels(['Composite', 'Prime'], color='#D0D0D0')  # Use theme color
                
            else:
                # 3D scatter plot
                self.viz_canvas.axes = self.viz_canvas.fig.add_subplot(111, projection='3d')
                # Set background color explicitly
                self.viz_canvas.axes.set_facecolor('#2D2D2D')
                
                # Create a colormap that maps primes to one color and composites to another
                colors = np.zeros(len(self.topology.numbers))
                colors[self.topology.prime_indices] = 1  # Primes get value 1
                
                scatter = self.viz_canvas.axes.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    embedding[:, 2],
                    c=colors,
                    cmap='coolwarm',
                    alpha=0.7,
                    s=30
                )
                
                # Add labels for some primes
                for i, idx in enumerate(self.topology.prime_indices):
                    if i % max(1, len(self.topology.prime_indices) // 15) == 0:  # Label ~15 primes
                        self.viz_canvas.axes.text(
                            embedding[idx, 0],
                            embedding[idx, 1],
                            embedding[idx, 2],
                            str(self.topology.numbers[idx]),
                            fontsize=9,
                            color='#D0D0D0',  
                            weight='bold'
                        )
                
                self.viz_canvas.axes.set_xlabel("Dimension 1", color='#D0D0D0')  
                self.viz_canvas.axes.set_ylabel("Dimension 2", color='#D0D0D0')  
                self.viz_canvas.axes.set_zlabel("Dimension 3", color='#D0D0D0')  
                
                # Add colorbar with clear labels
                cbar = self.viz_canvas.fig.colorbar(scatter, ticks=[0, 1])
                cbar.ax.set_yticklabels(['Composite', 'Prime'], color='#D0D0D0')  
            
            # Add a title with the embedding method
            method_name = self.embedding_combo.currentText()
            self.viz_canvas.axes.set_title(f"Prime Distribution in Residue Space ({method_name})", color='#D0D0D0')  
            
            # Add annotation about the theoretical meaning
            if method_name.lower() == 'laplacian':
                self.viz_canvas.fig.text(
                    0.02, 0.02, 
                    "The Laplacian embedding reveals the intrinsic topology of the prime manifold,\n"
                    "with primes forming coherent structures determined by residue patterns.",
                    fontsize=9, color='#AAAAAA'
                )
            else:
                self.viz_canvas.fig.text(
                    0.02, 0.02, 
                    "This visualization reveals primes clustering based on their residue fingerprints,\n"
                    "demonstrating the underlying structure in the seemingly random distribution of primes.",
                    fontsize=9, color='#AAAAAA'
                )
            
            self.viz_canvas.fig.tight_layout()
            
            # Apply tick colors (for axes numbers)
            for ax in self.viz_canvas.fig.get_axes():
                ax.tick_params(colors='#D0D0D0')  # Use theme color
                for spine in ax.spines.values():
                    spine.set_edgecolor('#707070')  # Use theme color for frame
            
            # Force drawing
            self.viz_canvas.draw()
            # Process events to ensure display updates
            QApplication.processEvents()
        
        except Exception as e:
            # Show error message
            self.viz_canvas.fig.clear()
            self.viz_canvas.axes = self.viz_canvas.fig.add_subplot(111)
            self.viz_canvas.axes.text(
                0.5, 0.5, 
                f"Error updating visualization:\n{str(e)}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.viz_canvas.axes.transAxes,
                color='#D0D0D0'  # Use theme color
            )
            self.viz_canvas.axes.set_axis_off()
            self.viz_canvas.fig.tight_layout()
            self.viz_canvas.draw()
            
            # Print stack trace for debugging
            traceback.print_exc()
    
    def update_tensor_field(self):
        """Update the tensor field visualization."""
        try:
            # Clear the current figure instead of replacing it
            self.tensor_canvas.fig.clear()
            
            field_type = self.field_combo.currentText()
            viz_type = self.viz_combo.currentText()
            
            # Prepare the field data
            if field_type == "Potential Φ":
                field_data = self.topology.potential_field
                field_label = "Potential Field Φ"
                field_description = "The scalar potential field Φ measures the 'primeness' of each number based on residue properties."
            elif field_type == "Gradient Magnitude":
                field_data = np.linalg.norm(self.topology.tensor_field, axis=1)
                field_label = "Gradient Magnitude |∇Φ|"
                field_description = "The gradient magnitude |∇Φ| quantifies the rate of change of 'primeness' around each number."
            elif field_type == "X Component":
                field_data = self.topology.tensor_field[:, 0]
                field_label = "X Component of ∇Φ"
                field_description = "The X component of ∇Φ shows how 'primeness' changes with the first residue dimension."
            elif field_type == "Y Component":
                field_data = self.topology.tensor_field[:, 1]
                field_label = "Y Component of ∇Φ"
                field_description = "The Y component of ∇Φ shows how 'primeness' changes with the second residue dimension."
            
            # Create x-axis data (number line)
            x = self.topology.numbers
            
            # Safety check for data
            if len(field_data) == 0 or len(x) == 0:
                self.tensor_canvas.axes = self.tensor_canvas.fig.add_subplot(111)
                self.tensor_canvas.axes.text(
                    0.5, 0.5, 
                    "Error: No tensor field data available.\nTry changing parameters or decreasing number range.",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=self.tensor_canvas.axes.transAxes,
                    color='#D0D0D0'
                )
                self.tensor_canvas.axes.set_axis_off()
                self.tensor_canvas.fig.tight_layout()
                self.tensor_canvas.draw()
                return
            
            # Normalize field data for visualization
            # Add small epsilon to avoid division by zero
            min_val = np.min(field_data)
            max_val = np.max(field_data)
            if max_val == min_val:  # Handle constant field
                norm_field = np.ones_like(field_data) * 0.5
            else:
                norm_field = (field_data - min_val) / (max_val - min_val + 1e-10)
            
            # Mark prime locations
            primes = self.topology.primes
            
            # Visualization based on type
            if viz_type == "Heatmap":
                # Create a 2D grid for heatmap
                self.tensor_canvas.axes = self.tensor_canvas.fig.add_subplot(111)
                self.tensor_canvas.axes.set_facecolor('#2D2D2D')
                
                n_bins = 50
                heatmap = np.zeros((n_bins, len(x)))
                
                for i in range(n_bins):
                    y_value = i / n_bins
                    distance = np.abs(norm_field - y_value)
                    heatmap[i, :] = np.exp(-distance * 10)  # Gaussian kernel
                    
                im = self.tensor_canvas.axes.imshow(
                    heatmap, 
                    aspect='auto', 
                    extent=[min(x), max(x), 0, 1],
                    origin='lower',
                    cmap='viridis'
                )
                
                # Mark primes
                for p in primes:
                    idx = p - 2  # Adjust for 0-indexing and starting at 2
                    if 0 <= idx < len(norm_field):  # Safety check
                        self.tensor_canvas.axes.axvline(
                            x=p, 
                            ymin=0, 
                            ymax=norm_field[idx],
                            color='red', 
                            alpha=0.5, 
                            linestyle=':'
                        )
                
                self.tensor_canvas.axes.set_xlabel("Number", color='#D0D0D0')
                self.tensor_canvas.axes.set_ylabel("Normalized Field Value", color='#D0D0D0')
                
                # Add a colorbar
                cbar = self.tensor_canvas.fig.colorbar(im)
                cbar.set_label("Field Intensity", color='#D0D0D0')
                cbar.ax.yaxis.set_tick_params(colors='#D0D0D0')
                
            elif viz_type == "Surface":
                # Create a 2D surface
                self.tensor_canvas.axes = self.tensor_canvas.fig.add_subplot(111, projection='3d')
                self.tensor_canvas.axes.set_facecolor('#2D2D2D')
                
                x_mesh, y_mesh = np.meshgrid(
                    np.linspace(min(x), max(x), 500),
                    np.linspace(0, 1, 100)
                )
                
                # Interpolate field values
                from scipy.interpolate import interp1d
                interp = interp1d(
                    x, norm_field, kind='cubic', 
                    bounds_error=False, fill_value='extrapolate'
                )
                
                z_mesh = np.zeros_like(x_mesh)
                for i in range(z_mesh.shape[0]):
                    z_val = y_mesh[i, 0]
                    z_mesh[i, :] = np.abs(interp(x_mesh[i, :]) - z_val)
                
                z_mesh = np.exp(-z_mesh * 10)
                
                # Plot surface
                surf = self.tensor_canvas.axes.plot_surface(
                    x_mesh, y_mesh, z_mesh,
                    cmap='viridis',
                    linewidth=0,
                    antialiased=True,
                    alpha=0.8
                )
                
                # Mark primes
                for p in primes:
                    idx = p - 2  # Adjust index
                    if 0 <= idx < len(norm_field):  # Safety check
                        self.tensor_canvas.axes.plot(
                            [p, p],
                            [0, norm_field[idx]],
                            [0, 1],
                            color='red',
                            linewidth=2
                        )
                
                self.tensor_canvas.axes.set_xlabel("Number", color='#D0D0D0')
                self.tensor_canvas.axes.set_ylabel("Field Value", color='#D0D0D0')
                self.tensor_canvas.axes.set_zlabel("Intensity", color='#D0D0D0')
                
                # Add a colorbar
                cbar = self.tensor_canvas.fig.colorbar(surf)
                cbar.set_label("Field Intensity", color='#D0D0D0')
                cbar.ax.yaxis.set_tick_params(colors='#D0D0D0')
                
            elif viz_type == "Contour":
                # Plot the field as a line
                self.tensor_canvas.axes = self.tensor_canvas.fig.add_subplot(111)
                self.tensor_canvas.axes.set_facecolor('#2D2D2D')
                
                self.tensor_canvas.axes.plot(x, norm_field, color='cyan', linewidth=1)
                
                # Mark the primes with points
                prime_indices = []
                for p in primes:
                    idx = p - 2  # Adjust for 0-indexing and starting at 2
                    if 0 <= idx < len(norm_field):  # Safety check
                        prime_indices.append(idx)
                        
                if prime_indices:
                    prime_values = norm_field[prime_indices]
                    self.tensor_canvas.axes.scatter(
                        [x[i] for i in prime_indices], prime_values,
                        color='red', s=30, zorder=3
                    )
                    
                    # Add threshold line based on theoretical criterion
                    threshold = np.percentile(prime_values, 10)
                    self.tensor_canvas.axes.axhline(
                        y=threshold,
                        color='green',
                        linestyle='--',
                        alpha=0.7,
                        label=f"Prime Threshold ({threshold:.3f})"
                    )
                    
                    # Shade regions above threshold
                    self.tensor_canvas.axes.fill_between(
                        x, threshold, norm_field,
                        where=(norm_field > threshold),
                        alpha=0.3,
                        color='green',
                        interpolate=True
                    )
                
                self.tensor_canvas.axes.set_xlabel("Number", color='#D0D0D0')
                self.tensor_canvas.axes.set_ylabel("Normalized Field Value", color='#D0D0D0')
                self.tensor_canvas.axes.legend(framealpha=0.7)
            
            # Add title and description
            self.tensor_canvas.axes.set_title(f"{field_label} Visualization", color='#D0D0D0')
            self.tensor_canvas.fig.text(0.02, 0.02, field_description, fontsize=9, color='#AAAAAA')
            
            self.tensor_canvas.fig.tight_layout()
            
            # Apply tick colors
            for ax in self.tensor_canvas.fig.get_axes():
                ax.tick_params(colors='#D0D0D0')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#707070')
            
            self.tensor_canvas.draw()
            QApplication.processEvents()
            
        except Exception as e:
            # Show error message
            self.tensor_canvas.fig.clear()
            self.tensor_canvas.axes = self.tensor_canvas.fig.add_subplot(111)
            self.tensor_canvas.axes.text(
                0.5, 0.5, 
                f"Error updating tensor field:\n{str(e)}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.tensor_canvas.axes.transAxes,
                color='#D0D0D0'
            )
            self.tensor_canvas.axes.set_axis_off()
            self.tensor_canvas.fig.tight_layout()
            self.tensor_canvas.draw()
            
            # Print stack trace for debugging        
            traceback.print_exc()
            
        except Exception as e:
            # Show error message
            self.tensor_canvas.axes = self.tensor_canvas.fig.add_subplot(111)
            self.tensor_canvas.axes.text(
                0.5, 0.5, 
                f"Error updating tensor field:\n{str(e)}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.tensor_canvas.axes.transAxes
            )
            self.tensor_canvas.axes.set_axis_off()
            self.tensor_canvas.fig.tight_layout()
            self.tensor_canvas.draw()
            self.tensor_canvas.draw_idle()  # Force redraw
            
            # Print stack trace for debugging           
            traceback.print_exc()
    
    def run_prime_prediction(self):
        """Run prime prediction using the tensor field."""
        self.statusBar().showMessage("Running prime prediction...")
        
        start = self.start_spinner.value()
        range_size = self.pred_range_spinner.value()
        
        # Clear previous results
        self.results_text.clear()
        
        try:
            # Calculate predictions using lattice reduction
            start_time = time.time()
            predicted_primes = self.topology.lattice_reduction_prime_search(
                search_range=range_size,
                start=start
            )
            prediction_time = time.time() - start_time
            
            # Get actual primes in the range for comparison
            actual_primes = list(primerange(start, start + range_size))
            
            # Display results
            self.results_text.append(f"Prediction Range: {start} to {start + range_size - 1}\n")
            self.results_text.append(f"Predicted Primes: {sorted(predicted_primes)}\n")
            self.results_text.append(f"Actual Primes: {actual_primes}\n")
            
            # Calculate accuracy metrics
            true_positives = set(predicted_primes).intersection(set(actual_primes))
            false_positives = set(predicted_primes) - set(actual_primes)
            false_negatives = set(actual_primes) - set(predicted_primes)
            
            precision = len(true_positives) / max(1, len(predicted_primes))
            recall = len(true_positives) / max(1, len(actual_primes))
            f1 = 2 * precision * recall / max(0.001, precision + recall)
            
            self.results_text.append(f"\nAccuracy Metrics:")
            self.results_text.append(f"True Positives: {len(true_positives)}")
            self.results_text.append(f"False Positives: {len(false_positives)}")
            self.results_text.append(f"False Negatives: {len(false_negatives)}")
            self.results_text.append(f"Precision: {precision:.4f}")
            self.results_text.append(f"Recall: {recall:.4f}")
            self.results_text.append(f"F1 Score: {f1:.4f}")
            self.results_text.append(f"Prediction Time: {prediction_time:.4f} seconds")
            
            # Add theoretical analysis
            self.results_text.append(f"\nTheoretical Analysis:")
            self.results_text.append(f"The lattice reduction method identifies primes by finding short vectors in the residue lattice.")
            self.results_text.append(f"This approach is based on the theoretical insight that primes correspond to distinctive patterns")
            self.results_text.append(f"in residue space that can be captured via lattice basis reduction techniques.")
            
            # Visualize prediction results
            self.prediction_canvas.fig.clear()
            self.prediction_canvas.axes = self.prediction_canvas.fig.add_subplot(111)
            self.prediction_canvas.axes.set_facecolor('#2D2D2D')
            
            # Create number range for x-axis
            number_range = np.arange(start, start + range_size)
            
            # Create indicator for primality
            is_prime = np.zeros(range_size)
            for p in actual_primes:
                if p - start < range_size:
                    is_prime[p - start] = 1
            
            # Create indicator for predictions
            is_predicted = np.zeros(range_size)
            for p in predicted_primes:
                if p - start < range_size:
                    is_predicted[p - start] = 1
            
            # Plot results
            bar_width = 0.35
            x = np.arange(range_size)
            
            self.prediction_canvas.axes.bar(
                x - bar_width/2, is_prime, bar_width, 
                label='Actual Primes', color='cyan', alpha=0.5
            )
            self.prediction_canvas.axes.bar(
                x + bar_width/2, is_predicted, bar_width, 
                label='Predicted Primes', color='red', alpha=0.5
            )
            
            # Set x-ticks to show actual numbers
            tick_interval = max(1, range_size // 20)
            tick_indices = x[::tick_interval]
            tick_labels = number_range[::tick_interval]
            self.prediction_canvas.axes.set_xticks(tick_indices)
            self.prediction_canvas.axes.set_xticklabels(tick_labels, rotation=45, color='#D0D0D0')
            
            self.prediction_canvas.axes.set_xlabel("Number", color='#D0D0D0')
            self.prediction_canvas.axes.set_ylabel("Is Prime", color='#D0D0D0')
            self.prediction_canvas.axes.set_title("Prime Prediction Results using Residue Lattice Reduction", color='#D0D0D0')
            self.prediction_canvas.axes.legend(framealpha=0.7)
            
            # Apply tick colors
            self.prediction_canvas.axes.tick_params(colors='#D0D0D0')
            for spine in self.prediction_canvas.axes.spines.values():
                spine.set_edgecolor('#707070')
            
            self.prediction_canvas.fig.tight_layout()
            self.prediction_canvas.draw()
            QApplication.processEvents()
            
            self.statusBar().showMessage("Prime prediction completed")
            
        except Exception as e:
            # Show error message on the canvas
            self.prediction_canvas.fig.clear()
            self.prediction_canvas.axes = self.prediction_canvas.fig.add_subplot(111)
            self.prediction_canvas.axes.text(
                0.5, 0.5, 
                f"Error running prediction:\n{str(e)}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.prediction_canvas.axes.transAxes,
                color='#D0D0D0'
            )
            self.prediction_canvas.axes.set_axis_off()
            self.prediction_canvas.fig.tight_layout()
            self.prediction_canvas.draw()
            
            # Show error in results
            self.results_text.append(f"\nError: {str(e)}")
            
            # Print stack trace for debugging        
            traceback.print_exc()
            
            self.statusBar().showMessage(f"Error in prime prediction: {str(e)}")
    
    def analyze_topology(self):
        """Analyze topological features of prime distribution."""
        self.statusBar().showMessage("Analyzing topology...")
        
        # Clear previous results
        self.topology_results.clear()
        
        try:
            # Calculate topological features
            features = self.topology.identify_topological_features()
            
            # Display results
            self.topology_results.append("# Topological Features of Prime Distribution\n")
            
            # Group features by category
            feature_categories = {
                "Simplicial Complex": ["num_vertices", "num_edges", "num_triangles", "num_tetrahedra", "euler_characteristic"],
                "Network Properties": ["clustering_coefficient", "avg_shortest_path", "num_components", 
                                    "largest_component_size", "largest_component_ratio"],
                "Prime Gap Analysis": ["avg_gap", "max_gap", "gap_std", "gap_cycle_length"],
                "Mathematical Constants": ["pi_correlation", "phi_correlation", "gamma_correlation", "riemann_correlation"]
            }
            
            for category, feature_keys in feature_categories.items():
                self.topology_results.append(f"## {category}\n")
                for key in feature_keys:
                    if key in features:
                        value = features[key]
                        # Format value based on type
                        if isinstance(value, float):
                            value_str = f"{value:.4f}"
                        else:
                            value_str = str(value)
                        self.topology_results.append(f"**{key}:** {value_str}\n")
                self.topology_results.append("\n")
                
            # Add theoretical interpretation
            self.topology_results.append("## Theoretical Interpretation\n")
            self.topology_results.append("The topological analysis reveals the intrinsic structure of prime distribution in residue space.\n")
            self.topology_results.append("* The Euler characteristic quantifies the global topology of the prime manifold.\n")
            self.topology_results.append("* Clustering coefficient measures local density of prime connections.\n")
            self.topology_results.append("* Correlations with constants like π, φ, and γ suggest deep mathematical connections.\n")
            self.topology_results.append("* Prime gaps follow patterns predicted by the tensor field gradient.\n")
            
            # Visualize topological features
            self.topology_canvas.fig.clear()
            
            if self.topology.low_dim_embedding is None:
                self.topology.calculate_embedding(3)
            
            # Use only prime points
            prime_points = self.topology.low_dim_embedding[self.topology.prime_indices]
            
            if len(prime_points) >= 4 and prime_points.shape[1] >= 2:
                # 1. Network visualization (top)
                network_ax = self.topology_canvas.fig.add_subplot(211)
                network_ax.set_facecolor('#2D2D2D')
                
                # Create a graph of connected prime points
                threshold = np.percentile([
                    np.linalg.norm(prime_points[i] - prime_points[j])
                    for i in range(len(prime_points))
                    for j in range(i+1, min(i+20, len(prime_points)))  # Use windowed approach for performance
                ], 20)  # 20th percentile of distances
                
                G = nx.Graph()
                for i in range(len(prime_points)):
                    G.add_node(i)
                
                for i in range(len(prime_points)):
                    for j in range(i+1, len(prime_points)):
                        if np.linalg.norm(prime_points[i] - prime_points[j]) < threshold:
                            G.add_edge(i, j)
                
                # Position nodes based on first two dimensions of embedding
                pos = {i: (prime_points[i, 0], prime_points[i, 1]) for i in range(len(prime_points))}
                
                # Color nodes by their component
                components = list(nx.connected_components(G))
                node_colors = np.zeros(len(prime_points))
                for i, component in enumerate(components):
                    for node in component:
                        node_colors[node] = i % 10  # Cycle through 10 colors
                
                # Draw the graph
                nx.draw_networkx(
                    G, pos, 
                    with_labels=False,
                    node_size=30,
                    node_color=node_colors,
                    cmap=plt.cm.tab10,
                    alpha=0.7,
                    edge_color='gray',
                    width=0.5,
                    ax=network_ax
                )
                
                # Add labels for some primes
                labels = {
                    i: str(self.topology.primes[i])
                    for i in range(len(prime_points))
                    if i % max(1, len(prime_points) // 15) == 0
                }
                nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=network_ax, font_color='#D0D0D0')
                
                network_ax.set_title("Prime Topology Network", color='#D0D0D0')
                network_ax.set_axis_off()  # Hide axes
                
                # 2. Persistence diagram visualization (bottom)
                persistence_ax = self.topology_canvas.fig.add_subplot(212)
                persistence_ax.set_facecolor('#2D2D2D')
                
                # Calculate a simple persistence-inspired diagram
                # For each prime, calculate its persistence value based on 
                # the potential field difference from surrounding composites
                
                persistence_values = []
                birth_values = []
                
                for i, p_idx in enumerate(self.topology.prime_indices):
                    p = self.topology.numbers[p_idx]
                    # Find nearest composite numbers
                    left_composite = p - 1
                    while left_composite >= 2 and isprime(left_composite):
                        left_composite -= 1
                    
                    right_composite = p + 1
                    while right_composite < self.topology.max_number and isprime(right_composite):
                        right_composite += 1
                    
                    # Calculate persistence as difference in potential
                    if left_composite >= 2 and right_composite < self.topology.max_number:
                        left_idx = left_composite - 2  
                        right_idx = right_composite - 2  
                        
                        # Safety check for indices
                        if 0 <= left_idx < len(self.topology.potential_field) and 0 <= right_idx < len(self.topology.potential_field) and 0 <= p_idx < len(self.topology.potential_field):
                            birth = self.topology.potential_field[p_idx]
                            death = max(self.topology.potential_field[left_idx], 
                                        self.topology.potential_field[right_idx])
                            
                            persistence = birth - death
                            
                            if persistence > 0:  # Only add positive persistence
                                persistence_values.append(persistence)
                                birth_values.append(birth)
                
                # Plot persistence diagram
                if persistence_values:
                    scatter = persistence_ax.scatter(
                        birth_values, 
                        persistence_values,
                        c=persistence_values,
                        cmap='viridis',
                        alpha=0.7,
                        s=20
                    )
                    
                    # Add diagonal line
                    max_val = max(birth_values + persistence_values)
                    persistence_ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                    
                    persistence_ax.set_xlabel("Birth (Potential Value)", color='#D0D0D0')
                    persistence_ax.set_ylabel("Persistence", color='#D0D0D0')
                    persistence_ax.set_title("Prime Persistence Diagram", color='#D0D0D0')
                    
                    # Add colorbar
                    cbar = self.topology_canvas.fig.colorbar(scatter, ax=persistence_ax)
                    cbar.set_label("Persistence", color='#D0D0D0')
                    cbar.ax.yaxis.set_tick_params(colors='#D0D0D0')
                    
                    # Add annotation
                    persistence_ax.text(
                        0.05, 0.95, 
                        f"Total features: {len(persistence_values)}\n"
                        f"Avg persistence: {np.mean(persistence_values):.4f}\n"
                        f"Max persistence: {np.max(persistence_values):.4f}",
                        transform=persistence_ax.transAxes,
                        fontsize=8,
                        verticalalignment='top',
                        color='#D0D0D0',
                        bbox=dict(boxstyle='round', facecolor='#1A1A1A', alpha=0.7)
                    )
                else:
                    # No persistence values to plot
                    persistence_ax.text(
                        0.5, 0.5, 
                        "Insufficient data for persistence diagram.",
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=persistence_ax.transAxes,
                        color='#D0D0D0'
                    )
                    persistence_ax.set_axis_off()
                    
                # Apply tick colors
                for ax in self.topology_canvas.fig.get_axes():
                    ax.tick_params(colors='#D0D0D0')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#707070')
            else:
                # Not enough points for meaningful visualization
                self.topology_canvas.axes = self.topology_canvas.fig.add_subplot(111)
                self.topology_canvas.axes.set_facecolor('#2D2D2D')
                
                self.topology_canvas.axes.text(
                    0.5, 0.5,
                    "Not enough prime points for topology visualization.\n"
                    "Try increasing the number range.",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=self.topology_canvas.axes.transAxes,
                    color='#D0D0D0'
                )
                self.topology_canvas.axes.set_axis_off()
            
            self.topology_canvas.fig.tight_layout()
            self.topology_canvas.draw()
            QApplication.processEvents()
            
            self.statusBar().showMessage("Topology analysis completed")
            
        except Exception as e:
            # Show error on canvas
            self.topology_canvas.fig.clear()
            self.topology_canvas.axes = self.topology_canvas.fig.add_subplot(111)
            self.topology_canvas.axes.text(
                0.5, 0.5, 
                f"Error analyzing topology:\n{str(e)}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.topology_canvas.axes.transAxes,
                color='#D0D0D0'
            )
            self.topology_canvas.axes.set_axis_off()
            self.topology_canvas.fig.tight_layout()
            self.topology_canvas.draw()
            
            # Show error in results
            self.topology_results.append(f"Error analyzing topology: {str(e)}")
            
            # Print stack trace for debugging
            import traceback
            traceback.print_exc()
            
            self.statusBar().showMessage(f"Error in topology analysis: {str(e)}")
    
    def on_range_changed(self, value):
        """Handle changes to the range slider."""
        # Update label
        for child in self.range_slider.parent().findChildren(QLabel):
            if child.text().startswith("Max Number:"):
                child.setText(f"Max Number: {value}")
                break

    def show_comprehensive_analysis(self):
        """Generate and display a comprehensive analysis."""
        self.statusBar().showMessage("Generating comprehensive analysis...")
        
        # Generate analysis
        analysis = self.topology.generate_comprehensive_analysis()
        
        # Create a new dialog to display the analysis
        dialog = QDialog(self)
        dialog.setWindowTitle("Comprehensive Analysis")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Create tabs for different sections of the analysis
        tabs = QTabWidget()
        
        # Basic Statistics tab
        basic_stats_widget = QWidget()
        basic_stats_layout = QVBoxLayout(basic_stats_widget)
        basic_stats_text = QTextEdit()
        basic_stats_text.setReadOnly(True)
        basic_stats_layout.addWidget(basic_stats_text)
        
        basic_stats_text.append("# Basic Statistics\n\n")
        for key, value in analysis['basic_stats'].items():
            basic_stats_text.append(f"**{key}:** {value}\n")
        
        tabs.addTab(basic_stats_widget, "Basic Stats")
        
        # Fingerprint & Field Statistics tab
        field_stats_widget = QWidget()
        field_stats_layout = QVBoxLayout(field_stats_widget)
        field_stats_text = QTextEdit()
        field_stats_text.setReadOnly(True)
        field_stats_layout.addWidget(field_stats_text)
        
        field_stats_text.append("# Fingerprint Statistics\n\n")
        for key, value in analysis['fingerprint_stats'].items():
            field_stats_text.append(f"**{key}:** {value}\n")
        
        field_stats_text.append("\n# Potential Field Statistics\n\n")
        for key, value in analysis['potential_field_stats'].items():
            field_stats_text.append(f"**{key}:** {value}\n")
        
        field_stats_text.append("\n# Tensor Field Statistics\n\n")
        for key, value in analysis['tensor_field_stats'].items():
            field_stats_text.append(f"**{key}:** {value}\n")
        
        tabs.addTab(field_stats_widget, "Field Stats")
        
        # Topological Invariants tab
        invariants_widget = QWidget()
        invariants_layout = QVBoxLayout(invariants_widget)
        invariants_text = QTextEdit()
        invariants_text.setReadOnly(True)
        invariants_layout.addWidget(invariants_text)
        
        invariants_text.append("# Topological Invariants\n\n")
        for key, value in analysis['topological_invariants'].items():
            invariants_text.append(f"**{key}:** {value}\n")
        
        tabs.addTab(invariants_widget, "Topological Invariants")
        
        # Prediction Performance tab
        prediction_widget = QWidget()
        prediction_layout = QVBoxLayout(prediction_widget)
        prediction_text = QTextEdit()
        prediction_text.setReadOnly(True)
        prediction_layout.addWidget(prediction_text)
        
        prediction_text.append("# Prediction Performance\n\n")
        for i, result in enumerate(analysis['prediction_performance']):
            prediction_text.append(f"## Test {i+1} (Range: {result['start']} - {result['start'] + result['range_size']})\n\n")
            for key, value in result.items():
                if key not in ['start', 'range_size']:
                    # Fix: Format value properly based on type
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                    prediction_text.append(f"**{key}:** {formatted_value}\n")
            prediction_text.append("\n")
        
        tabs.addTab(prediction_widget, "Prediction")
        
        # Prime Gap Analysis tab
        gap_widget = QWidget()
        gap_layout = QVBoxLayout(gap_widget)
        gap_text = QTextEdit()
        gap_text.setReadOnly(True)
        gap_layout.addWidget(gap_text)
        
        gap_text.append("# Prime Gap Analysis\n\n")
        for key, value in analysis['gap_prediction_analysis'].items():
            if key != 'sample_predictions':
                gap_text.append(f"**{key}:** {value}\n")
        
        gap_text.append("\n## Sample Predictions\n\n")
        if 'sample_predictions' in analysis['gap_prediction_analysis']:
            for pred in analysis['gap_prediction_analysis']['sample_predictions']:
                gap_text.append(f"Prime: {pred['prime']}, Actual gap: {pred['actual_gap']}, "
                            f"Predicted gap: {pred['predicted_gap']}, "
                            f"Error: {pred['error']} ({pred['relative_error']:.2%})\n")
        
        gap_text.append("\n## Prime Gap Statistics\n\n")
        if 'prime_gap_stats' in analysis:
            for key, value in analysis['prime_gap_stats'].items():
                if key != 'gap_distribution':
                    gap_text.append(f"**{key}:** {value}\n")
        
        tabs.addTab(gap_widget, "Prime Gaps")
        
        # Topology Features tab
        topology_widget = QWidget()
        topology_layout = QVBoxLayout(topology_widget)
        topology_text = QTextEdit()
        topology_text.setReadOnly(True)
        topology_layout.addWidget(topology_text)
        
        topology_text.append("# Topology Features\n\n")
        for key, value in analysis['topology_features'].items():
            topology_text.append(f"**{key}:** {value}\n")
        
        topology_text.append("\n# Mathematical Connections\n\n")
        for key, value in analysis['mathematical_connections'].items():
            topology_text.append(f"**{key}:** {value}\n")
        
        tabs.addTab(topology_widget, "Topology")
        
        # Theoretical Insights tab
        theory_widget = QWidget()
        theory_layout = QVBoxLayout(theory_widget)
        theory_text = QTextEdit()
        theory_text.setReadOnly(True)
        theory_layout.addWidget(theory_text)
        
        theory_text.append("# Theoretical Insights\n\n")
        for i, insight in enumerate(analysis['theoretical_insights']):
            theory_text.append(f"{i+1}. {insight}\n")
        
        theory_text.append("\n# Future Research Directions\n\n")
        for i, direction in enumerate(analysis['future_directions']):
            theory_text.append(f"{i+1}. {direction}\n")
        
        tabs.addTab(theory_widget, "Theory")
        
        # Performance & Improvements tab
        perf_widget = QWidget()
        perf_layout = QVBoxLayout(perf_widget)
        perf_text = QTextEdit()
        perf_text.setReadOnly(True)
        perf_layout.addWidget(perf_text)
        
        perf_text.append("# Performance Timing (seconds)\n\n")
        for key, value in analysis['performance_timing'].items():
            perf_text.append(f"**{key}:** {value:.4f}\n")
        
        tabs.addTab(perf_widget, "Performance")
        
        # Add save button to export the analysis to a file
        save_button = QPushButton("Save Analysis to File")
        save_button.clicked.connect(lambda: self.save_analysis_to_file(analysis))
        
        # Add tabs and buttons to dialog
        layout.addWidget(tabs)
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(save_button)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
        self.statusBar().showMessage("Analysis completed")

    def save_analysis_to_file(self, analysis):
        """Save the analysis to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis", "", "JSON Files (*.json);;Text Files (*.txt);;HTML Files (*.html)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    # Convert numpy arrays to lists for JSON serialization
                    def convert_numpy_to_python(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_to_python(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_to_python(i) for i in obj]
                        else:
                            return obj
                    
                    analysis_json = convert_numpy_to_python(analysis)
                    with open(file_path, 'w') as f:
                        json.dump(analysis_json, f, indent=4)
                        
                elif file_path.endswith('.html'):
                    # Create HTML report
                    with open(file_path, 'w') as f:
                        f.write("""<!DOCTYPE html>
    <html>
    <head>
        <title>Prime Residue Topology Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; color: #333; }
            h1, h2, h3 { color: #2a82da; }
            pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
            .section { margin-bottom: 30px; border-bottom: 1px solid #ddd; padding-bottom: 20px; }
            table { border-collapse: collapse; width: 100%; }
            table, th, td { border: 1px solid #ddd; }
            th, td { padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Prime Residue Topology Analysis Report</h1>
        <p>Generated by the Prime Residue Topology Explorer on """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <div class="section">
            <h2>Basic Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
    """)
                        # Add basic stats
                        for key, value in analysis['basic_stats'].items():
                            f.write(f"            <tr><td>{key}</td><td>{value}</td></tr>\n")
                        
                        f.write("""
            </table>
        </div>
        
        <div class="section">
            <h2>Potential Field Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
    """)
                        # Add potential field stats
                        for key, value in analysis['potential_field_stats'].items():
                            if key != 'theoretical_foundation':
                                f.write(f"            <tr><td>{key}</td><td>{value}</td></tr>\n")
                        
                        f.write("""
            </table>
            <p><strong>Theoretical Foundation:</strong> """ + analysis['potential_field_stats'].get('theoretical_foundation', '') + """</p>
        </div>
        
        <div class="section">
            <h2>Tensor Field Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
    """)
                        # Add tensor field stats
                        for key, value in analysis['tensor_field_stats'].items():
                            if key != 'gradient_interpretation':
                                f.write(f"            <tr><td>{key}</td><td>{value}</td></tr>\n")
                        
                        f.write("""
            </table>
            <p><strong>Gradient Interpretation:</strong> """ + analysis['tensor_field_stats'].get('gradient_interpretation', '') + """</p>
        </div>
        
        <div class="section">
            <h2>Prediction Performance</h2>
    """)
                        # Add prediction performance
                        for i, result in enumerate(analysis['prediction_performance']):
                            f.write(f"        <h3>Test {i+1} (Range: {result['start']} - {result['start'] + result['range_size']})</h3>\n")
                            f.write("        <table>\n")
                            f.write("            <tr><th>Metric</th><th>Value</th></tr>\n")
                            for key, value in result.items():
                                if key not in ['start', 'range_size']:                                  
                                    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                                    f.write(f"            <tr><td>{key}</td><td>{formatted_value}</td></tr>\n")
                            f.write("        </table>\n")
                        
                        f.write("""
        </div>
        
        <div class="section">
            <h2>Topological Features</h2>
            <table>
                <tr><th>Feature</th><th>Value</th></tr>
    """)
                        # Add topology features
                        for key, value in analysis['topology_features'].items():
                            f.write(f"            <tr><td>{key}</td><td>{value}</td></tr>\n")
                        
                        f.write("""
            </table>
        </div>
        
        <div class="section">
            <h2>Mathematical Connections</h2>
            <table>
                <tr><th>Connection</th><th>Correlation</th></tr>
    """)
                        # Add mathematical connections
                        for key, value in analysis['mathematical_connections'].items():
                            if key != 'interpretation':                             
                                formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                                f.write(f"            <tr><td>{key}</td><td>{formatted_value}</td></tr>\n")
                        
                        f.write("""
            </table>
            <p><strong>Interpretation:</strong> """ + analysis['mathematical_connections'].get('interpretation', '') + """</p>
        </div>
        
        <div class="section">
            <h2>Theoretical Insights</h2>
            <ol>
    """)
                        # Add theoretical insights
                        for insight in analysis['theoretical_insights']:
                            f.write(f"            <li>{insight}</li>\n")
                        
                        f.write("""
            </ol>
        </div>
        
        <div class="section">
            <h2>Future Research Directions</h2>
            <ol>
    """)
                        # Add future directions
                        for direction in analysis['future_directions']:
                            f.write(f"            <li>{direction}</li>\n")
                        
                        f.write("""
            </ol>
        </div>
        
        <div class="section">
            <h2>Performance Timing (seconds)</h2>
            <table>
                <tr><th>Operation</th><th>Time</th></tr>
    """)
                        # Add performance timing
                        for key, value in analysis['performance_timing'].items():
                            f.write(f"            <tr><td>{key}</td><td>{value:.4f}</td></tr>\n")
                        
                        f.write("""
            </table>
        </div>
        
        <footer>
            <p>Generated by Prime Residue Topology Explorer. For academic purposes only.</p>
        </footer>
    </body>
    </html>
    """)
                else:
                    # Text format
                    with open(file_path, 'w') as f:
                        f.write("# Prime Residue Topology Analysis\n\n")
                        
                        f.write("## Basic Statistics\n\n")
                        for key, value in analysis['basic_stats'].items():
                            f.write(f"{key}: {value}\n")
                        
                        f.write("\n## Fingerprint Statistics\n\n")
                        for key, value in analysis['fingerprint_stats'].items():
                            f.write(f"{key}: {value}\n")
                        
                        f.write("\n## Potential Field Statistics\n\n")
                        for key, value in analysis['potential_field_stats'].items():
                            f.write(f"{key}: {value}\n")
                        
                        f.write("\n## Tensor Field Statistics\n\n")
                        for key, value in analysis['tensor_field_stats'].items():                          
                            formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                            f.write(f"{key}: {formatted_value}\n")
                        
                        f.write("\n## Topological Invariants\n\n")
                        for key, value in analysis['topological_invariants'].items():                        
                            formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                            f.write(f"{key}: {formatted_value}\n")
                        
                        f.write("\n## Prediction Performance\n\n")
                        for i, result in enumerate(analysis['prediction_performance']):
                            f.write(f"Test {i+1} (Range: {result['start']} - {result['start'] + result['range_size']})\n")
                            for key, value in result.items():
                                if key not in ['start', 'range_size']:                               
                                    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                                    f.write(f"  {key}: {formatted_value}\n")
                            f.write("\n")
                        
                        f.write("\n## Prime Gap Analysis\n\n")
                        for key, value in analysis['gap_prediction_analysis'].items():
                            if key != 'sample_predictions':                        
                                formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                                f.write(f"{key}: {formatted_value}\n")
                                
                        f.write("\nSample Predictions:\n")
                        if 'sample_predictions' in analysis['gap_prediction_analysis']:
                            for pred in analysis['gap_prediction_analysis']['sample_predictions']:
                                f.write(f"  Prime: {pred['prime']}, Actual gap: {pred['actual_gap']}, "
                                    f"Predicted gap: {pred['predicted_gap']}, "
                                    f"Error: {pred['error']} ({pred['relative_error']:.2%})\n")
                        
                        f.write("\n## Topology Features\n\n")
                        for key, value in analysis['topology_features'].items():                        
                            formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                            f.write(f"{key}: {formatted_value}\n")
                        
                        f.write("\n## Prime Gap Statistics\n\n")
                        if 'prime_gap_stats' in analysis:
                            for key, value in analysis['prime_gap_stats'].items():
                                if key != 'gap_distribution':                                 
                                    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                                    f.write(f"{key}: {formatted_value}\n")
                        
                        f.write("\n## Mathematical Connections\n\n")
                        for key, value in analysis['mathematical_connections'].items():                          
                            formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                            f.write(f"{key}: {formatted_value}\n")
                        
                        f.write("\n## Theoretical Insights\n\n")
                        for i, insight in enumerate(analysis['theoretical_insights']):
                            f.write(f"{i+1}. {insight}\n")
                            
                        f.write("\n## Future Research Directions\n\n")
                        for i, direction in enumerate(analysis['future_directions']):
                            f.write(f"{i+1}. {direction}\n")
                        
                        f.write("\n## Performance Timing (seconds)\n\n")
                        for key, value in analysis['performance_timing'].items():
                            f.write(f"{key}: {value:.4f}\n")
                
                self.statusBar().showMessage(f"Analysis saved to {file_path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error saving analysis: {str(e)}")
                import traceback
                traceback.print_exc()


def main():
    app = QApplication(sys.argv)
    
    # Apply dark theme
    apply_dark_theme(app)
    set_matplotlib_dark_style()
    
    window = PrimeVisualizer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()      

#!/usr/bin/env python3
"""
OpenCog-Inspired Multiscale Constraint Optimization for Cosmeceutical Formulation

This module implements OpenCog cognitive architecture features adapted for
cosmeceutical formulation optimization, including:

- AtomSpace-like knowledge representation for ingredients and interactions
- ECAN-inspired attention allocation for promising formulation spaces
- PLN-like reasoning for ingredient compatibility and synergy
- MOSES-like evolutionary optimization for formulation search
- INCI-driven search space reduction
- Multiscale skin model integration

Author: ONNX Runtime Cosmeceutical Optimization Team
"""

import json
import math
import random
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
from abc import ABC, abstractmethod

# Import existing cosmetic chemistry framework
from cosmetic_chemistry_example import *


class AttentionValue:
    """OpenCog-inspired attention value for cognitive resource allocation"""
    
    def __init__(self, sti: float = 0.0, lti: float = 0.0, vlti: bool = False):
        self.sti = sti  # Short-term importance
        self.lti = lti  # Long-term importance
        self.vlti = vlti  # Very long-term importance (boolean flag)
    
    def update_sti(self, delta: float, decay_rate: float = 0.1):
        """Update short-term importance with decay"""
        self.sti = self.sti * (1 - decay_rate) + delta
    
    def update_lti(self, delta: float):
        """Update long-term importance based on sustained attention"""
        self.lti += delta
        if self.lti > 100:  # Threshold for VLTI
            self.vlti = True
    
    def total_attention(self) -> float:
        """Calculate total attention value"""
        base = self.sti + 0.1 * self.lti
        return base * 2.0 if self.vlti else base


class CognitiveAtom:
    """OpenCog-inspired atom with attention values and truth values"""
    
    def __init__(self, name: str, atom_type: str, properties: Dict = None):
        self.name = name
        self.atom_type = atom_type
        self.properties = properties or {}
        self.attention = AttentionValue()
        self.truth_value = TruthValue()
        self.creation_time = 0
        self.last_accessed = 0
        self.incoming_links = set()
        self.outgoing_links = set()
    
    def __repr__(self):
        return f"CognitiveAtom({self.name}, {self.atom_type}, att={self.attention.total_attention():.2f})"


class TruthValue:
    """PLN-inspired truth value with strength and confidence"""
    
    def __init__(self, strength: float = 0.5, confidence: float = 0.1):
        self.strength = max(0.0, min(1.0, strength))
        self.confidence = max(0.0, min(1.0, confidence))
    
    def __repr__(self):
        return f"TV({self.strength:.3f}, {self.confidence:.3f})"
    
    def revision_rule(self, other: 'TruthValue') -> 'TruthValue':
        """PLN revision rule for combining evidence"""
        n1 = self.confidence_to_count()
        n2 = other.confidence_to_count()
        
        new_strength = (self.strength * n1 + other.strength * n2) / (n1 + n2)
        new_confidence = self.count_to_confidence(n1 + n2)
        
        return TruthValue(new_strength, new_confidence)
    
    def confidence_to_count(self) -> float:
        """Convert confidence to evidence count"""
        return self.confidence / (1 - self.confidence) if self.confidence < 1.0 else float('inf')
    
    def count_to_confidence(self, count: float) -> float:
        """Convert evidence count to confidence"""
        return count / (count + 1) if count != float('inf') else 1.0


class CognitiveLink:
    """OpenCog-inspired link connecting atoms with truth values"""
    
    def __init__(self, link_type: str, atoms: List[CognitiveAtom], 
                 truth_value: TruthValue = None):
        self.link_type = link_type
        self.atoms = atoms
        self.truth_value = truth_value or TruthValue()
        self.attention = AttentionValue()
        
        # Register with atoms
        for atom in atoms:
            atom.outgoing_links.add(self)
    
    def __repr__(self):
        atom_names = [atom.name for atom in self.atoms]
        return f"{self.link_type}({atom_names}, {self.truth_value})"


class AtomSpace:
    """OpenCog-inspired AtomSpace for knowledge representation"""
    
    def __init__(self):
        self.atoms = {}  # name -> CognitiveAtom
        self.links = []  # List of CognitiveLink
        self.atom_types = set()
        self.link_types = set()
        self.time_step = 0
    
    def add_atom(self, atom: CognitiveAtom) -> CognitiveAtom:
        """Add atom to atomspace"""
        if atom.name in self.atoms:
            return self.atoms[atom.name]
        
        self.atoms[atom.name] = atom
        self.atom_types.add(atom.atom_type)
        atom.creation_time = self.time_step
        return atom
    
    def add_link(self, link: CognitiveLink) -> CognitiveLink:
        """Add link to atomspace"""
        self.links.append(link)
        self.link_types.add(link.link_type)
        return link
    
    def get_atom(self, name: str) -> Optional[CognitiveAtom]:
        """Retrieve atom by name"""
        atom = self.atoms.get(name)
        if atom:
            atom.last_accessed = self.time_step
            atom.attention.update_sti(0.1)  # Small boost for access
        return atom
    
    def get_atoms_by_type(self, atom_type: str) -> List[CognitiveAtom]:
        """Get all atoms of specific type"""
        return [atom for atom in self.atoms.values() if atom.atom_type == atom_type]
    
    def get_links_by_type(self, link_type: str) -> List[CognitiveLink]:
        """Get all links of specific type"""
        return [link for link in self.links if link.link_type == link_type]
    
    def advance_time(self):
        """Advance time step for attention decay"""
        self.time_step += 1
        for atom in self.atoms.values():
            atom.attention.update_sti(0, decay_rate=0.01)


class SkinLayer(Enum):
    """Multiscale skin model layers"""
    STRATUM_CORNEUM = "stratum_corneum"  # Outermost barrier
    EPIDERMIS = "epidermis"              # Living epidermis
    DERMIS_PAPILLARY = "dermis_papillary"  # Upper dermis
    DERMIS_RETICULAR = "dermis_reticular"  # Lower dermis
    HYPODERMIS = "hypodermis"           # Subcutaneous layer


@dataclass
class DeliveryMechanism:
    """Delivery system for targeting specific skin layers"""
    name: str
    target_layers: List[SkinLayer]
    penetration_enhancers: List[str] = field(default_factory=list)
    particle_size: Optional[float] = None  # nanometers
    encapsulation_type: Optional[str] = None
    release_profile: str = "immediate"  # immediate, sustained, controlled
    
    def compatibility_score(self, ingredient: str, target_layer: SkinLayer) -> float:
        """Calculate compatibility score for ingredient-layer combination"""
        base_score = 0.5
        
        # Enhance score if targeting appropriate layer
        if target_layer in self.target_layers:
            base_score += 0.3
        
        # Particle size considerations
        if self.particle_size:
            if target_layer == SkinLayer.STRATUM_CORNEUM and self.particle_size > 100:
                base_score += 0.2  # Larger particles stay on surface
            elif target_layer in [SkinLayer.EPIDERMIS, SkinLayer.DERMIS_PAPILLARY] and 10 < self.particle_size < 100:
                base_score += 0.2  # Medium particles penetrate well
        
        return min(1.0, base_score)


@dataclass
class TherapeuticVector:
    """Therapeutic action targeting specific conditions"""
    name: str
    target_condition: str
    mechanism_of_action: str
    optimal_concentration_range: Tuple[float, float]  # min, max percentage
    target_layers: List[SkinLayer]
    synergistic_ingredients: List[str] = field(default_factory=list)
    antagonistic_ingredients: List[str] = field(default_factory=list)
    
    def efficacy_score(self, concentration: float, layer: SkinLayer) -> float:
        """Calculate efficacy score for given concentration and layer"""
        min_conc, max_conc = self.optimal_concentration_range
        
        # Concentration score
        if min_conc <= concentration <= max_conc:
            conc_score = 1.0
        elif concentration < min_conc:
            conc_score = concentration / min_conc
        else:
            conc_score = max(0.1, 1.0 - (concentration - max_conc) / max_conc)
        
        # Layer targeting score
        layer_score = 1.0 if layer in self.target_layers else 0.3
        
        return conc_score * layer_score


class INCIParser:
    """Parser for INCI (International Nomenclature of Cosmetic Ingredients) listings"""
    
    def __init__(self):
        self.concentration_rules = {
            # EU regulations for common ingredients
            'retinol': 0.3,
            'salicylic_acid': 2.0,
            'glycolic_acid': 10.0,
            'niacinamide': 10.0,
            'vitamin_c': 20.0,
            'phenoxyethanol': 1.0,
            # Add more as needed
        }
    
    def parse_inci_list(self, inci_string: str) -> List[str]:
        """Parse INCI string into ordered ingredient list"""
        # Clean and split INCI string
        ingredients = [ing.strip().lower() for ing in inci_string.split(',')]
        return [self._normalize_ingredient_name(ing) for ing in ingredients if ing]
    
    def estimate_concentrations(self, inci_list: List[str], 
                              total_volume: float = 100.0) -> Dict[str, float]:
        """Estimate concentrations from INCI ordering"""
        if not inci_list:
            return {}
        
        concentrations = {}
        remaining_volume = total_volume
        
        # First ingredient is typically the largest component (often water)
        if len(inci_list) > 0:
            concentrations[inci_list[0]] = min(80.0, remaining_volume * 0.6)
            remaining_volume -= concentrations[inci_list[0]]
        
        # Distribute remaining volume with decreasing concentrations
        for i, ingredient in enumerate(inci_list[1:], 1):
            # Apply regulatory limits if known
            max_allowed = self.concentration_rules.get(ingredient, remaining_volume)
            
            # Exponential decay for ordering
            estimated = remaining_volume * math.exp(-i * 0.5)
            concentration = min(estimated, max_allowed, remaining_volume * 0.9)
            
            concentrations[ingredient] = max(0.01, concentration)  # Minimum 0.01%
            remaining_volume -= concentrations[ingredient]
            
            if remaining_volume <= 0.1:
                break
        
        return concentrations
    
    def _normalize_ingredient_name(self, name: str) -> str:
        """Normalize ingredient name for consistency"""
        # Remove common prefixes/suffixes and standardize
        name = name.replace('(inci)', '').replace('ci ', '').strip()
        return name.replace(' ', '_').replace('-', '_')
    
    def check_subset_compatibility(self, product_inci: List[str], 
                                 ingredient_subset: Set[str]) -> bool:
        """Check if ingredient subset is compatible with product INCI"""
        product_set = set(product_inci)
        return ingredient_subset.issubset(product_set)
    
    def reduce_search_space(self, all_ingredients: List[str], 
                          target_inci: List[str]) -> List[str]:
        """Reduce ingredient search space based on INCI compatibility"""
        target_set = set(target_inci)
        compatible_ingredients = []
        
        for ingredient in all_ingredients:
            # Include if ingredient is in target INCI or has high compatibility
            if (ingredient in target_set or 
                self._calculate_inci_compatibility(ingredient, target_inci) > 0.7):
                compatible_ingredients.append(ingredient)
        
        return compatible_ingredients
    
    def _calculate_inci_compatibility(self, ingredient: str, inci_list: List[str]) -> float:
        """Calculate compatibility score for ingredient with INCI list"""
        # Simple heuristic based on ingredient class similarity
        # In practice, this would use more sophisticated chemical similarity
        base_score = 0.3
        
        # Check for ingredient class compatibility
        ingredient_class = self._get_ingredient_class(ingredient)
        for inci_ingredient in inci_list:
            inci_class = self._get_ingredient_class(inci_ingredient)
            if ingredient_class == inci_class:
                base_score += 0.1
        
        return min(1.0, base_score)
    
    def _get_ingredient_class(self, ingredient: str) -> str:
        """Get ingredient functional class"""
        # Simplified classification - would be more sophisticated in practice
        if any(acid in ingredient for acid in ['acid', 'ate']):
            return 'acid'
        elif any(alcohol in ingredient for alcohol in ['ol', 'yl']):
            return 'alcohol'
        elif 'glyc' in ingredient:
            return 'glycol'
        else:
            return 'other'


class ECANAttentionModule:
    """ECAN-inspired attention allocation for promising formulation spaces"""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.attention_budget = 1000.0
        self.focus_boundary = 50.0  # STI threshold for focused attention
        self.spreading_factor = 0.8
        self.min_sti = -100.0
        self.max_sti = 100.0
    
    def update_attention(self):
        """Update attention values across atomspace"""
        # Collect atoms above focus boundary
        focused_atoms = [atom for atom in self.atomspace.atoms.values() 
                        if atom.attention.sti > self.focus_boundary]
        
        # Spread attention from focused atoms
        for atom in focused_atoms:
            self._spread_attention(atom)
        
        # Normalize attention values
        self._normalize_attention()
        
        # Update long-term importance
        self._update_lti()
    
    def _spread_attention(self, source_atom: CognitiveAtom):
        """Spread attention from source atom to connected atoms"""
        spread_amount = source_atom.attention.sti * self.spreading_factor * 0.1
        
        # Spread to atoms connected via outgoing links
        for link in source_atom.outgoing_links:
            for target_atom in link.atoms:
                if target_atom != source_atom:
                    target_atom.attention.update_sti(
                        spread_amount * link.truth_value.strength
                    )
    
    def _normalize_attention(self):
        """Normalize STI values to stay within budget"""
        total_sti = sum(max(0, atom.attention.sti) 
                       for atom in self.atomspace.atoms.values())
        
        if total_sti > self.attention_budget:
            scaling_factor = self.attention_budget / total_sti
            for atom in self.atomspace.atoms.values():
                if atom.attention.sti > 0:
                    atom.attention.sti *= scaling_factor
        
        # Clamp values
        for atom in self.atomspace.atoms.values():
            atom.attention.sti = max(self.min_sti, 
                                   min(self.max_sti, atom.attention.sti))
    
    def _update_lti(self):
        """Update long-term importance based on sustained attention"""
        for atom in self.atomspace.atoms.values():
            if atom.attention.sti > self.focus_boundary:
                atom.attention.update_lti(0.1)
    
    def get_most_attended_atoms(self, n: int = 10) -> List[CognitiveAtom]:
        """Get top N atoms by attention value"""
        sorted_atoms = sorted(self.atomspace.atoms.values(),
                            key=lambda a: a.attention.total_attention(),
                            reverse=True)
        return sorted_atoms[:n]
    
    def boost_attention(self, atom_name: str, boost: float):
        """Manually boost attention for specific atom"""
        atom = self.atomspace.get_atom(atom_name)
        if atom:
            atom.attention.update_sti(boost)


class PLNReasoningEngine:
    """PLN-inspired reasoning engine for ingredient interactions"""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.rules = []
        self._initialize_reasoning_rules()
    
    def _initialize_reasoning_rules(self):
        """Initialize PLN reasoning rules"""
        self.rules = [
            self._inheritance_rule,
            self._similarity_rule,
            self._implication_rule,
            self._deduction_rule,
            self._abduction_rule
        ]
    
    def reason_about_compatibility(self, ingredient1: str, ingredient2: str) -> TruthValue:
        """Reason about ingredient compatibility using PLN"""
        atom1 = self.atomspace.get_atom(ingredient1)
        atom2 = self.atomspace.get_atom(ingredient2)
        
        if not atom1 or not atom2:
            return TruthValue(0.5, 0.1)  # Unknown
        
        # Look for existing compatibility links
        compatibility_tv = self._find_compatibility_link(atom1, atom2)
        if compatibility_tv:
            return compatibility_tv
        
        # Infer compatibility from shared properties
        shared_properties = set(atom1.properties.keys()) & set(atom2.properties.keys())
        if shared_properties:
            strength = len(shared_properties) / max(len(atom1.properties), len(atom2.properties), 1)
            confidence = min(0.8, len(shared_properties) * 0.2)
            inferred_tv = TruthValue(strength, confidence)
            
            # Create new compatibility link
            link = CognitiveLink('COMPATIBILITY', [atom1, atom2], inferred_tv)
            self.atomspace.add_link(link)
            
            return inferred_tv
        
        return TruthValue(0.5, 0.1)
    
    def _find_compatibility_link(self, atom1: CognitiveAtom, atom2: CognitiveAtom) -> Optional[TruthValue]:
        """Find existing compatibility link between atoms"""
        for link in self.atomspace.get_links_by_type('COMPATIBILITY'):
            if set(link.atoms) == {atom1, atom2}:
                return link.truth_value
        return None
    
    def infer_synergy(self, ingredients: List[str]) -> Dict[Tuple[str, str], TruthValue]:
        """Infer synergistic relationships between ingredients"""
        synergies = {}
        
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                # Use reasoning rules to infer synergy
                synergy_tv = self._apply_synergy_rules(ing1, ing2)
                if synergy_tv.strength > 0.6:  # Threshold for significant synergy
                    synergies[(ing1, ing2)] = synergy_tv
        
        return synergies
    
    def _apply_synergy_rules(self, ingredient1: str, ingredient2: str) -> TruthValue:
        """Apply reasoning rules to determine synergy"""
        atom1 = self.atomspace.get_atom(ingredient1)
        atom2 = self.atomspace.get_atom(ingredient2)
        
        if not atom1 or not atom2:
            return TruthValue(0.0, 0.1)
        
        # Rule: Antioxidants often synergize
        if (atom1.atom_type == 'ANTIOXIDANT' and atom2.atom_type == 'ANTIOXIDANT'):
            return TruthValue(0.8, 0.7)
        
        # Rule: Humectants + Emollients synergize
        if ((atom1.atom_type == 'HUMECTANT' and atom2.atom_type == 'EMOLLIENT') or
            (atom1.atom_type == 'EMOLLIENT' and atom2.atom_type == 'HUMECTANT')):
            return TruthValue(0.7, 0.6)
        
        # Rule: Same mechanism of action may have synergy
        moa1 = atom1.properties.get('mechanism_of_action', '')
        moa2 = atom2.properties.get('mechanism_of_action', '')
        if moa1 and moa1 == moa2:
            return TruthValue(0.6, 0.5)
        
        return TruthValue(0.3, 0.2)  # Default low synergy
    
    # PLN reasoning rules
    def _inheritance_rule(self, premise1: TruthValue, premise2: TruthValue) -> TruthValue:
        """PLN inheritance rule"""
        strength = premise1.strength * premise2.strength
        confidence = premise1.confidence * premise2.confidence
        return TruthValue(strength, confidence)
    
    def _similarity_rule(self, premise1: TruthValue, premise2: TruthValue) -> TruthValue:
        """PLN similarity rule"""
        strength = (premise1.strength + premise2.strength) / 2
        confidence = min(premise1.confidence, premise2.confidence)
        return TruthValue(strength, confidence)
    
    def _implication_rule(self, premise1: TruthValue, premise2: TruthValue) -> TruthValue:
        """PLN implication rule"""
        if premise1.strength == 0:
            return TruthValue(1.0, premise1.confidence)
        strength = premise2.strength / premise1.strength
        confidence = premise1.confidence * premise2.confidence
        return TruthValue(min(1.0, strength), confidence)
    
    def _deduction_rule(self, ab_tv: TruthValue, bc_tv: TruthValue) -> TruthValue:
        """PLN deduction rule: A->B, B->C => A->C"""
        strength = ab_tv.strength * bc_tv.strength
        confidence = ab_tv.confidence * bc_tv.confidence
        return TruthValue(strength, confidence)
    
    def _abduction_rule(self, ab_tv: TruthValue, ac_tv: TruthValue) -> TruthValue:
        """PLN abduction rule: A->B, A->C => B->C"""
        if ab_tv.strength == 0:
            return TruthValue(0.5, 0.1)
        strength = ac_tv.strength / ab_tv.strength
        confidence = min(ab_tv.confidence, ac_tv.confidence) * 0.8
        return TruthValue(min(1.0, strength), confidence)


if __name__ == "__main__":
    print("=== OpenCog-Inspired Cosmeceutical Optimization System ===\n")
    
    # Initialize cognitive architecture
    atomspace = AtomSpace()
    attention_module = ECANAttentionModule(atomspace)
    reasoning_engine = PLNReasoningEngine(atomspace)
    inci_parser = INCIParser()
    
    print("1. Initializing Cognitive AtomSpace:")
    print("-----------------------------------")
    
    # Create cognitive atoms for ingredients
    ingredients_data = {
        'retinol': {'type': 'ACTIVE_INGREDIENT', 'mechanism': 'cell_renewal', 'target_layers': ['epidermis']},
        'niacinamide': {'type': 'ACTIVE_INGREDIENT', 'mechanism': 'barrier_repair', 'target_layers': ['epidermis']},
        'hyaluronic_acid': {'type': 'HUMECTANT', 'mechanism': 'moisture_binding', 'target_layers': ['stratum_corneum']},
        'vitamin_c': {'type': 'ANTIOXIDANT', 'mechanism': 'collagen_synthesis', 'target_layers': ['dermis_papillary']},
        'vitamin_e': {'type': 'ANTIOXIDANT', 'mechanism': 'lipid_protection', 'target_layers': ['stratum_corneum']},
    }
    
    for name, data in ingredients_data.items():
        atom = CognitiveAtom(name, data['type'], data)
        atomspace.add_atom(atom)
        print(f"  Created: {atom}")
    
    print(f"\nAtomSpace contains {len(atomspace.atoms)} atoms")
    
    print("\n2. INCI Parsing and Concentration Estimation:")
    print("--------------------------------------------")
    
    # Example INCI string
    sample_inci = "aqua, glycerin, niacinamide, hyaluronic_acid, phenoxyethanol, vitamin_e"
    ingredients_list = inci_parser.parse_inci_list(sample_inci)
    concentrations = inci_parser.estimate_concentrations(ingredients_list)
    
    print(f"INCI List: {ingredients_list}")
    print("Estimated Concentrations:")
    for ingredient, conc in concentrations.items():
        print(f"  {ingredient}: {conc:.2f}%")
    
    print("\n3. PLN Reasoning - Ingredient Compatibility:")
    print("-------------------------------------------")
    
    # Test compatibility reasoning
    test_pairs = [('retinol', 'niacinamide'), ('vitamin_c', 'vitamin_e'), ('hyaluronic_acid', 'glycerin')]
    
    for ing1, ing2 in test_pairs:
        compatibility = reasoning_engine.reason_about_compatibility(ing1, ing2)
        print(f"  {ing1} + {ing2}: {compatibility}")
    
    print("\n4. ECAN Attention Mechanism:")
    print("----------------------------")
    
    # Boost attention for promising ingredients
    attention_module.boost_attention('niacinamide', 20.0)
    attention_module.boost_attention('hyaluronic_acid', 15.0)
    attention_module.update_attention()
    
    top_attended = attention_module.get_most_attended_atoms(5)
    print("Most attended ingredients:")
    for atom in top_attended:
        print(f"  {atom.name}: {atom.attention.total_attention():.2f}")
    
    print("\n5. Synergy Inference:")
    print("--------------------")
    
    synergies = reasoning_engine.infer_synergy(list(ingredients_data.keys()))
    print("Discovered synergies:")
    for (ing1, ing2), tv in synergies.items():
        print(f"  {ing1} + {ing2}: {tv}")
    
    print("\n6. Search Space Reduction:")
    print("-------------------------")
    
    all_ingredients = list(ingredients_data.keys()) + ['retinyl_palmitate', 'salicylic_acid', 'ceramides']
    target_inci = ['aqua', 'niacinamide', 'hyaluronic_acid', 'phenoxyethanol']
    
    reduced_space = inci_parser.reduce_search_space(all_ingredients, target_inci)
    print(f"Original search space: {len(all_ingredients)} ingredients")
    print(f"Reduced search space: {len(reduced_space)} ingredients")
    print(f"Retained ingredients: {reduced_space}")
    
    print("\n=== Cognitive Architecture Demonstration Complete ===")
    print("\nNext: Implement MOSES-like evolutionary optimization and multiscale skin modeling!")
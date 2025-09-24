#!/usr/bin/env python3
"""
Cosmetic Chemistry Introduction Example

This example demonstrates the basic usage of cosmetic-specific atom types
within the ONNX Runtime cheminformatics framework. It covers fundamental
concepts including ingredient definition, basic formulation creation, and
simple compatibility analysis.

Author: ONNX Runtime Cosmetic Chemistry Team
"""

# Import the cosmetic chemistry atom types
# Note: In a real implementation, these would be imported from the ONNX Runtime
# cheminformatics module. For this example, we'll define simplified classes.

class CosmeticAtom:
    """Base class for cosmetic chemistry atoms"""
    def __init__(self, name, properties=None):
        self.name = name
        self.properties = properties or {}
    
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"

class CosmeticLink:
    """Base class for cosmetic chemistry links"""
    def __init__(self, atom1, atom2, strength=1.0):
        self.atom1 = atom1
        self.atom2 = atom2
        self.strength = strength
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.atom1}, {self.atom2})"

# Ingredient Category Atom Types
class ACTIVE_INGREDIENT(CosmeticAtom):
    """Active pharmaceutical/cosmetic ingredients"""
    pass

class PRESERVATIVE(CosmeticAtom):
    """Antimicrobial preservatives"""
    pass

class EMULSIFIER(CosmeticAtom):
    """Emulsification agents"""
    pass

class HUMECTANT(CosmeticAtom):
    """Moisture-binding agents"""
    pass

class ANTIOXIDANT(CosmeticAtom):
    """Oxidation inhibitors"""
    pass

# Formulation Type Atom Types
class SKINCARE_FORMULATION(CosmeticAtom):
    """Face and body care products"""
    def __init__(self, name, *ingredients):
        super().__init__(name)
        self.ingredients = list(ingredients)
    
    def add_ingredient(self, ingredient):
        """Add an ingredient to the formulation"""
        self.ingredients.append(ingredient)
    
    def __repr__(self):
        ingredient_names = [ing.name for ing in self.ingredients]
        return f"SKINCARE_FORMULATION('{self.name}', ingredients={ingredient_names})"

# Interaction Type Link Classes
class COMPATIBILITY_LINK(CosmeticLink):
    """Compatible ingredient pairs"""
    pass

class INCOMPATIBILITY_LINK(CosmeticLink):  
    """Incompatible ingredient pairs"""
    pass

class SYNERGY_LINK(CosmeticLink):
    """Synergistic interactions"""
    pass

def main():
    """Main example demonstrating basic cosmetic chemistry concepts"""
    
    print("=== Cosmetic Chemistry Introduction Example ===\n")
    
    # 1. Define Basic Cosmetic Ingredients
    print("1. Defining Basic Cosmetic Ingredients:")
    print("-" * 40)
    
    # Active ingredients
    hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid', {
        'molecular_weight': 'high',
        'solubility': 'water_soluble',
        'function': 'hydration'
    })
    
    niacinamide = ACTIVE_INGREDIENT('niacinamide', {
        'concentration_limit': '10%',
        'ph_stability': '5.0-7.0',
        'function': 'pore_minimizing'
    })
    
    vitamin_c = ACTIVE_INGREDIENT('vitamin_c_l_ascorbic_acid', {
        'concentration_limit': '20%',
        'ph_stability': '3.0-4.0',
        'function': 'antioxidant_brightening'
    })
    
    # Base ingredients
    glycerin = HUMECTANT('glycerin', {
        'concentration_range': '1-10%',
        'solubility': 'water_soluble',
        'viscosity_impact': 'increases'
    })
    
    phenoxyethanol = PRESERVATIVE('phenoxyethanol', {
        'concentration_limit': '1%',
        'spectrum': 'broad_spectrum',
        'ph_range': '3-10'
    })
    
    cetyl_alcohol = EMULSIFIER('cetyl_alcohol', {
        'emulsion_type': 'w_o',
        'melting_point': '49-51C',
        'function': 'thickening_emulsifying'
    })
    
    vitamin_e = ANTIOXIDANT('vitamin_e_tocopherol', {
        'concentration_range': '0.1-1%',
        'function': 'lipid_antioxidant',
        'synergy': 'vitamin_c'
    })
    
    print(f"Hyaluronic Acid: {hyaluronic_acid}")
    print(f"Niacinamide: {niacinamide}")
    print(f"Vitamin C: {vitamin_c}")
    print(f"Glycerin: {glycerin}")
    print(f"Phenoxyethanol: {phenoxyethanol}")
    print(f"Cetyl Alcohol: {cetyl_alcohol}")
    print(f"Vitamin E: {vitamin_e}")
    
    # 2. Create Simple Formulations
    print("\n2. Creating Simple Formulations:")
    print("-" * 40)
    
    # Hydrating serum formulation
    hydrating_serum = SKINCARE_FORMULATION(
        'hydrating_serum',
        hyaluronic_acid,
        glycerin,
        phenoxyethanol
    )
    
    # Anti-aging moisturizer formulation
    anti_aging_moisturizer = SKINCARE_FORMULATION(
        'anti_aging_moisturizer',
        niacinamide,
        cetyl_alcohol,
        glycerin,
        phenoxyethanol
    )
    
    print(f"Hydrating Serum: {hydrating_serum}")
    print(f"Anti-Aging Moisturizer: {anti_aging_moisturizer}")
    
    # 3. Ingredient Compatibility Analysis
    print("\n3. Ingredient Compatibility Analysis:")
    print("-" * 40)
    
    # Define compatibility relationships
    compatibility_hyaluronic_niacinamide = COMPATIBILITY_LINK(
        hyaluronic_acid, niacinamide, strength=0.9
    )
    
    compatibility_glycerin_hyaluronic = COMPATIBILITY_LINK(
        glycerin, hyaluronic_acid, strength=0.8
    )
    
    # Define synergistic relationships
    synergy_vitamin_c_e = SYNERGY_LINK(
        vitamin_c, vitamin_e, strength=0.95
    )
    
    # Define incompatibility (common example)
    incompatibility_vitamin_c_niacinamide = INCOMPATIBILITY_LINK(
        vitamin_c, niacinamide, strength=0.3
    )
    
    print(f"Compatible: {compatibility_hyaluronic_niacinamide}")
    print(f"Compatible: {compatibility_glycerin_hyaluronic}")
    print(f"Synergistic: {synergy_vitamin_c_e}")
    print(f"Potentially Incompatible: {incompatibility_vitamin_c_niacinamide}")
    
    # 4. Formulation Analysis
    print("\n4. Formulation Analysis:")
    print("-" * 40)
    
    def analyze_formulation(formulation):
        """Analyze a formulation for basic properties"""
        print(f"\nAnalyzing: {formulation.name}")
        print(f"Ingredients count: {len(formulation.ingredients)}")
        
        # Categorize ingredients
        actives = [ing for ing in formulation.ingredients if isinstance(ing, ACTIVE_INGREDIENT)]
        preservatives = [ing for ing in formulation.ingredients if isinstance(ing, PRESERVATIVE)]
        emulsifiers = [ing for ing in formulation.ingredients if isinstance(ing, EMULSIFIER)]
        humectants = [ing for ing in formulation.ingredients if isinstance(ing, HUMECTANT)]
        
        print(f"Active ingredients: {len(actives)} - {[a.name for a in actives]}")
        print(f"Preservatives: {len(preservatives)} - {[p.name for p in preservatives]}")
        print(f"Emulsifiers: {len(emulsifiers)} - {[e.name for e in emulsifiers]}")
        print(f"Humectants: {len(humectants)} - {[h.name for h in humectants]}")
        
        # Basic formulation completeness check
        has_preservative = len(preservatives) > 0
        has_active = len(actives) > 0
        
        print(f"Formulation completeness:")
        print(f"  - Has active ingredient: {'✓' if has_active else '✗'}")
        print(f"  - Has preservative system: {'✓' if has_preservative else '✗'}")
    
    analyze_formulation(hydrating_serum)
    analyze_formulation(anti_aging_moisturizer)
    
    # 5. Compatibility Checking Function
    print("\n5. Compatibility Checking:")
    print("-" * 40)
    
    def check_ingredient_compatibility(ingredient1, ingredient2, known_links):
        """Check if two ingredients are compatible based on known links"""
        for link in known_links:
            if ((link.atom1.name == ingredient1.name and link.atom2.name == ingredient2.name) or
                (link.atom1.name == ingredient2.name and link.atom2.name == ingredient1.name)):
                return link
        return None
    
    # Example compatibility checks
    known_links = [
        compatibility_hyaluronic_niacinamide,
        compatibility_glycerin_hyaluronic,
        synergy_vitamin_c_e,
        incompatibility_vitamin_c_niacinamide
    ]
    
    test_pairs = [
        (hyaluronic_acid, niacinamide),
        (vitamin_c, vitamin_e),
        (vitamin_c, niacinamide),
        (glycerin, hyaluronic_acid)
    ]
    
    for ing1, ing2 in test_pairs:
        result = check_ingredient_compatibility(ing1, ing2, known_links)
        if result:
            link_type = result.__class__.__name__.replace('_LINK', '').lower()
            print(f"{ing1.name} + {ing2.name}: {link_type.capitalize()} (strength: {result.strength})")
        else:
            print(f"{ing1.name} + {ing2.name}: No known interaction")
    
    # 6. Advanced Formulation Creation
    print("\n6. Advanced Formulation Creation:")
    print("-" * 40)
    
    # Create a more complex vitamin C serum
    vitamin_c_serum = SKINCARE_FORMULATION('vitamin_c_antioxidant_serum')
    vitamin_c_serum.add_ingredient(vitamin_c)
    vitamin_c_serum.add_ingredient(vitamin_e)  # Synergistic antioxidant
    vitamin_c_serum.add_ingredient(glycerin)   # Hydrating base
    vitamin_c_serum.add_ingredient(phenoxyethanol)  # Preservation
    
    print(f"Advanced Formulation: {vitamin_c_serum}")
    analyze_formulation(vitamin_c_serum)
    
    print("\n=== Example Complete ===")
    print("\nThis example demonstrated:")
    print("• Basic cosmetic ingredient definition using atom types")
    print("• Simple formulation creation and ingredient combination")
    print("• Compatibility analysis using link types") 
    print("• Formulation analysis and completeness checking")
    print("• Advanced formulation building techniques")
    print("\nNext steps: See cosmetic_chemistry_example.py for advanced analysis!")

if __name__ == "__main__":
    main()
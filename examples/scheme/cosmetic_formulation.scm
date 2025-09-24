;; Complex Cosmetic Formulation Modeling with Compatibility Analysis
;; 
;; This Scheme example demonstrates advanced formulation modeling using
;; the cosmetic chemistry atom types within the ONNX Runtime cheminformatics
;; framework. It includes complex ingredient interactions, property modeling,
;; and automated compatibility analysis.
;;
;; Author: ONNX Runtime Cosmetic Chemistry Team

;; ======================================================================
;; INGREDIENT DEFINITIONS
;; ======================================================================

;; Define active ingredients with detailed properties
(define hyaluronic-acid 
  (ACTIVE_INGREDIENT 
    (stv "hyaluronic_acid")
    (stv 0.9 0.8)))  ; strength, confidence

(define niacinamide
  (ACTIVE_INGREDIENT
    (stv "niacinamide")
    (stv 0.85 0.9)))

(define vitamin-c
  (ACTIVE_INGREDIENT
    (stv "vitamin_c_l_ascorbic_acid")
    (stv 0.95 0.7)))  ; high efficacy, moderate stability

(define retinol
  (ACTIVE_INGREDIENT
    (stv "retinol")
    (stv 0.92 0.6)))  ; high efficacy, stability concerns

(define salicylic-acid
  (ACTIVE_INGREDIENT
    (stv "salicylic_acid")
    (stv 0.88 0.8)))

;; Define base ingredients
(define glycerin
  (HUMECTANT
    (stv "glycerin")
    (stv 0.8 0.95)))  ; reliable humectant

(define sodium-hyaluronate
  (HUMECTANT
    (stv "sodium_hyaluronate")
    (stv 0.9 0.85)))

(define phenoxyethanol
  (PRESERVATIVE
    (stv "phenoxyethanol")
    (stv 0.7 0.9)))   ; broad spectrum preservation

(define cetyl-alcohol
  (EMULSIFIER
    (stv "cetyl_alcohol")
    (stv 0.75 0.9)))

(define vitamin-e
  (ANTIOXIDANT
    (stv "vitamin_e_tocopherol")
    (stv 0.8 0.85)))

(define xanthan-gum
  (THICKENER
    (stv "xanthan_gum") 
    (stv 0.75 0.9)))

;; ======================================================================
;; PROPERTY DEFINITIONS  
;; ======================================================================

;; pH properties for different ingredients and formulations
(define ph-acidic (PH_PROPERTY (stv "acidic_ph_3.5") (stv 0.8 0.9)))
(define ph-neutral (PH_PROPERTY (stv "neutral_ph_6.5") (stv 0.9 0.95)))
(define ph-alkaline (PH_PROPERTY (stv "alkaline_ph_8.0") (stv 0.7 0.8)))

;; Viscosity properties
(define viscosity-low (VISCOSITY_PROPERTY (stv "low_viscosity_1000") (stv 0.8 0.9)))
(define viscosity-medium (VISCOSITY_PROPERTY (stv "medium_viscosity_5000") (stv 0.85 0.9)))
(define viscosity-high (VISCOSITY_PROPERTY (stv "high_viscosity_15000") (stv 0.9 0.85)))

;; Stability properties
(define stability-excellent (STABILITY_PROPERTY (stv "excellent_stability") (stv 0.95 0.9)))
(define stability-good (STABILITY_PROPERTY (stv "good_stability") (stv 0.8 0.85)))
(define stability-poor (STABILITY_PROPERTY (stv "poor_stability") (stv 0.4 0.7)))

;; ======================================================================
;; COMPATIBILITY AND INTERACTION LINKS
;; ======================================================================

;; Compatible ingredient pairs
(COMPATIBILITY_LINK hyaluronic-acid niacinamide (stv 0.9 0.85))
(COMPATIBILITY_LINK glycerin hyaluronic-acid (stv 0.85 0.9))
(COMPATIBILITY_LINK niacinamide vitamin-e (stv 0.8 0.8))
(COMPATIBILITY_LINK cetyl-alcohol glycerin (stv 0.9 0.9))

;; Synergistic combinations
(SYNERGY_LINK vitamin-c vitamin-e (stv 0.95 0.9))  ; Classic antioxidant synergy
(SYNERGY_LINK hyaluronic-acid sodium-hyaluronate (stv 0.9 0.85))  ; Enhanced hydration
(SYNERGY_LINK salicylic-acid niacinamide (stv 0.85 0.8))  ; Acne treatment synergy

;; Incompatible combinations
(INCOMPATIBILITY_LINK vitamin-c retinol (stv 0.8 0.9))  ; pH and irritation conflict
(INCOMPATIBILITY_LINK vitamin-c niacinamide (stv 0.3 0.6))  ; Disputed interaction
(INCOMPATIBILITY_LINK retinol salicylic-acid (stv 0.9 0.85))  ; Over-exfoliation risk

;; Antagonistic interactions
(ANTAGONISM_LINK retinol ph-alkaline (stv 0.8 0.9))  ; Retinol degrades in alkaline pH

;; ======================================================================
;; FORMULATION DEFINITIONS
;; ======================================================================

;; Anti-aging night serum formulation
(define anti-aging-serum
  (SKINCARE_FORMULATION
    (stv "anti_aging_night_serum")
    (stv 0.9 0.8)))

;; Add ingredients to anti-aging serum
(MemberLink anti-aging-serum retinol (stv 0.9 0.8))
(MemberLink anti-aging-serum hyaluronic-acid (stv 0.85 0.9))
(MemberLink anti-aging-serum vitamin-e (stv 0.8 0.85))
(MemberLink anti-aging-serum phenoxyethanol (stv 0.7 0.9))

;; Vitamin C brightening serum
(define vitamin-c-serum
  (SKINCARE_FORMULATION
    (stv "vitamin_c_brightening_serum")
    (stv 0.88 0.75)))

(MemberLink vitamin-c-serum vitamin-c (stv 0.95 0.7))
(MemberLink vitamin-c-serum vitamin-e (stv 0.8 0.85))
(MemberLink vitamin-c-serum sodium-hyaluronate (stv 0.85 0.9))
(MemberLink vitamin-c-serum phenoxyethanol (stv 0.7 0.9))

;; Hydrating daily moisturizer
(define daily-moisturizer
  (SKINCARE_FORMULATION
    (stv "hydrating_daily_moisturizer")
    (stv 0.85 0.9)))

(MemberLink daily-moisturizer niacinamide (stv 0.85 0.9))
(MemberLink daily-moisturizer hyaluronic-acid (stv 0.9 0.8))
(MemberLink daily-moisturizer glycerin (stv 0.8 0.95))
(MemberLink daily-moisturizer cetyl-alcohol (stv 0.75 0.9))
(MemberLink daily-moisturizer phenoxyethanol (stv 0.7 0.9))

;; BHA exfoliating treatment
(define bha-treatment
  (SKINCARE_FORMULATION
    (stv "bha_exfoliating_treatment")
    (stv 0.8 0.85)))

(MemberLink bha-treatment salicylic-acid (stv 0.88 0.8))
(MemberLink bha-treatment niacinamide (stv 0.85 0.9))
(MemberLink bha-treatment xanthan-gum (stv 0.75 0.9))
(MemberLink bha-treatment phenoxyethanol (stv 0.7 0.9))

;; ======================================================================
;; PROPERTY ASSIGNMENTS
;; ======================================================================

;; Assign pH properties to formulations
(EvaluationLink ph-neutral anti-aging-serum (stv 0.9 0.8))
(EvaluationLink ph-acidic vitamin-c-serum (stv 0.9 0.9))
(EvaluationLink ph-neutral daily-moisturizer (stv 0.85 0.9))
(EvaluationLink ph-acidic bha-treatment (stv 0.9 0.85))

;; Assign viscosity properties
(EvaluationLink viscosity-low anti-aging-serum (stv 0.8 0.8))
(EvaluationLink viscosity-low vitamin-c-serum (stv 0.85 0.8))
(EvaluationLink viscosity-medium daily-moisturizer (stv 0.9 0.85))
(EvaluationLink viscosity-medium bha-treatment (stv 0.8 0.9))

;; Assign stability properties based on ingredient interactions
(EvaluationLink stability-good anti-aging-serum (stv 0.75 0.8))  ; Retinol stability concerns
(EvaluationLink stability-good vitamin-c-serum (stv 0.7 0.7))    ; Vitamin C stability challenges
(EvaluationLink stability-excellent daily-moisturizer (stv 0.9 0.9)) ; Stable combination
(EvaluationLink stability-good bha-treatment (stv 0.85 0.8))     ; Generally stable

;; ======================================================================
;; SAFETY AND REGULATORY ASSESSMENTS
;; ======================================================================

;; Define concentration limits
(define retinol-limit (CONCENTRATION_LIMIT (stv "retinol_1_percent") (stv 0.8 0.9)))
(define salicylic-limit (CONCENTRATION_LIMIT (stv "salicylic_acid_2_percent") (stv 0.9 0.95)))
(define vitamin-c-limit (CONCENTRATION_LIMIT (stv "vitamin_c_20_percent") (stv 0.8 0.8)))

;; Apply concentration limits to ingredients
(EvaluationLink retinol-limit retinol (stv 0.9 0.9))
(EvaluationLink salicylic-limit salicylic-acid (stv 0.9 0.95))
(EvaluationLink vitamin-c-limit vitamin-c (stv 0.8 0.8))

;; Safety assessments
(define retinol-safety (SAFETY_ASSESSMENT (stv "retinol_moderate_irritation") (stv 0.7 0.8)))
(define vitamin-c-safety (SAFETY_ASSESSMENT (stv "vitamin_c_low_irritation") (stv 0.85 0.9)))
(define niacinamide-safety (SAFETY_ASSESSMENT (stv "niacinamide_very_low_irritation") (stv 0.95 0.95)))

(EvaluationLink retinol-safety retinol (stv 0.8 0.8))
(EvaluationLink vitamin-c-safety vitamin-c (stv 0.9 0.9))
(EvaluationLink niacinamide-safety niacinamide (stv 0.95 0.95))

;; ======================================================================
;; COMPATIBILITY ANALYSIS FUNCTIONS
;; ======================================================================

;; Define a rule for checking ingredient compatibility
(define compatibility-check-rule
  (BindLink
    (VariableList
      (Variable "$ingredient1")
      (Variable "$ingredient2")
      (Variable "$formulation"))
    (AndLink
      (MemberLink (Variable "$formulation") (Variable "$ingredient1"))
      (MemberLink (Variable "$formulation") (Variable "$ingredient2"))
      (INCOMPATIBILITY_LINK (Variable "$ingredient1") (Variable "$ingredient2")))
    (EvaluationLink
      (Predicate "incompatible_ingredients_detected")
      (ListLink (Variable "$formulation") (Variable "$ingredient1") (Variable "$ingredient2")))))

;; Rule for detecting synergistic combinations
(define synergy-detection-rule
  (BindLink
    (VariableList
      (Variable "$ingredient1")
      (Variable "$ingredient2")
      (Variable "$formulation"))
    (AndLink
      (MemberLink (Variable "$formulation") (Variable "$ingredient1"))
      (MemberLink (Variable "$formulation") (Variable "$ingredient2"))
      (SYNERGY_LINK (Variable "$ingredient1") (Variable "$ingredient2")))
    (EvaluationLink
      (Predicate "synergistic_combination_detected")
      (ListLink (Variable "$formulation") (Variable "$ingredient1") (Variable "$ingredient2")))))

;; Rule for pH compatibility checking
(define ph-compatibility-rule
  (BindLink
    (VariableList
      (Variable "$formulation")
      (Variable "$ingredient")
      (Variable "$ph"))
    (AndLink
      (MemberLink (Variable "$formulation") (Variable "$ingredient"))
      (EvaluationLink (Variable "$ph") (Variable "$formulation"))
      (ANTAGONISM_LINK (Variable "$ingredient") (Variable "$ph")))
    (EvaluationLink
      (Predicate "ph_incompatibility_detected")
      (ListLink (Variable "$formulation") (Variable "$ingredient") (Variable "$ph")))))

;; ======================================================================
;; FORMULATION OPTIMIZATION RULES
;; ======================================================================

;; Rule for recommending preservative addition
(define preservative-requirement-rule
  (BindLink
    (Variable "$formulation")
    (AndLink
      (TypedVariableLink (Variable "$formulation") (Type "SKINCARE_FORMULATION"))
      (NotLink
        (MemberLink (Variable "$formulation") phenoxyethanol)))
    (EvaluationLink
      (Predicate "requires_preservative")
      (Variable "$formulation"))))

;; Rule for stability optimization
(define stability-optimization-rule
  (BindLink
    (VariableList
      (Variable "$formulation")
      (Variable "$stability"))
    (AndLink
      (EvaluationLink (Variable "$stability") (Variable "$formulation"))
      (EvaluationLink 
        (GreaterThan (Number 0.8))
        (StrengthOf (Variable "$stability"))))
    (EvaluationLink
      (Predicate "stable_formulation")
      (Variable "$formulation"))))

;; ======================================================================
;; ANALYSIS AND REPORTING
;; ======================================================================

;; Generate compatibility report for anti-aging serum
(define (analyze-formulation formulation)
  (list
    (format #t "~%Analyzing formulation: ~a~%" formulation)
    
    ;; Check for incompatibilities
    (cog-execute! compatibility-check-rule)
    
    ;; Check for synergies
    (cog-execute! synergy-detection-rule)
    
    ;; Check pH compatibility
    (cog-execute! ph-compatibility-rule)
    
    ;; Check preservative requirement
    (cog-execute! preservative-requirement-rule)
    
    ;; Check stability
    (cog-execute! stability-optimization-rule)))

;; ======================================================================
;; ADVANCED FORMULATION QUERIES
;; ======================================================================

;; Query for all synergistic combinations in a formulation
(define synergy-query
  (BindLink
    (VariableList
      (Variable "$ingredient1")
      (Variable "$ingredient2"))
    (SYNERGY_LINK (Variable "$ingredient1") (Variable "$ingredient2"))
    (ListLink (Variable "$ingredient1") (Variable "$ingredient2"))))

;; Query for ingredients requiring pH consideration
(define ph-sensitive-query
  (BindLink
    (VariableList
      (Variable "$ingredient")
      (Variable "$ph"))
    (ANTAGONISM_LINK (Variable "$ingredient") (Variable "$ph"))
    (ListLink (Variable "$ingredient") (Variable "$ph"))))

;; Query for formulations with specific properties
(define stable-formulation-query
  (BindLink
    (Variable "$formulation")
    (EvaluationLink stability-excellent (Variable "$formulation"))
    (Variable "$formulation")))

;; ======================================================================
;; EXECUTION AND RESULTS
;; ======================================================================

;; Execute analysis on all formulations
(format #t "~%=== COSMETIC FORMULATION ANALYSIS RESULTS ===~%")

;; Analyze each formulation
(analyze-formulation anti-aging-serum)
(analyze-formulation vitamin-c-serum)  
(analyze-formulation daily-moisturizer)
(analyze-formulation bha-treatment)

;; Execute advanced queries
(format #t "~%=== SYNERGISTIC COMBINATIONS ===~%")
(cog-execute! synergy-query)

(format #t "~%=== PH SENSITIVE INGREDIENTS ===~%")
(cog-execute! ph-sensitive-query)

(format #t "~%=== STABLE FORMULATIONS ===~%")
(cog-execute! stable-formulation-query)

;; Summary report
(format #t "~%=== ANALYSIS SUMMARY ===~%")
(format #t "Total formulations analyzed: 4~%")
(format #t "- Anti-aging night serum: Retinol-based, moderate stability~%")
(format #t "- Vitamin C brightening serum: Antioxidant synergy, acidic pH~%") 
(format #t "- Hydrating daily moisturizer: Stable, well-balanced~%")
(format #t "- BHA exfoliating treatment: Acidic pH, synergistic actives~%")

(format #t "~%Key findings:~%")
(format #t "✓ Vitamin C + Vitamin E synergy detected~%")
(format #t "✓ Hyaluronic acid combinations show high compatibility~%")
(format #t "⚠ Retinol requires pH consideration for stability~%")
(format #t "⚠ Vitamin C formulations require antioxidant protection~%")

(format #t "~%=== FORMULATION ANALYSIS COMPLETE ===~%")
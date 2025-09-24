;; Simple Cosmetic Ingredient Compatibility Checking
;;
;; This Scheme example demonstrates basic ingredient interaction checking
;; using the cosmetic chemistry atom types. It provides a simplified
;; framework for analyzing ingredient compatibility in cosmetic formulations.
;;
;; Author: ONNX Runtime Cosmetic Chemistry Team

;; ======================================================================
;; BASIC INGREDIENT DEFINITIONS
;; ======================================================================

;; Define core active ingredients
(define retinol 
  (ACTIVE_INGREDIENT 
    (stv "retinol")
    (stv 0.9 0.8)))

(define vitamin-c
  (ACTIVE_INGREDIENT
    (stv "vitamin_c")
    (stv 0.95 0.7)))

(define niacinamide
  (ACTIVE_INGREDIENT
    (stv "niacinamide") 
    (stv 0.85 0.9)))

(define salicylic-acid
  (ACTIVE_INGREDIENT
    (stv "salicylic_acid")
    (stv 0.88 0.8)))

(define hyaluronic-acid
  (ACTIVE_INGREDIENT
    (stv "hyaluronic_acid")
    (stv 0.9 0.85)))

;; Define supporting ingredients
(define vitamin-e
  (ANTIOXIDANT
    (stv "vitamin_e")
    (stv 0.8 0.85)))

(define glycerin
  (HUMECTANT
    (stv "glycerin")
    (stv 0.8 0.95)))

;; ======================================================================
;; COMPATIBILITY DEFINITIONS
;; ======================================================================

;; Compatible combinations (safe to use together)
(COMPATIBILITY_LINK hyaluronic-acid niacinamide (stv 0.9 0.85))
(COMPATIBILITY_LINK niacinamide vitamin-e (stv 0.8 0.8))
(COMPATIBILITY_LINK hyaluronic-acid glycerin (stv 0.85 0.9))
(COMPATIBILITY_LINK glycerin vitamin-e (stv 0.75 0.8))

;; Synergistic combinations (work better together)
(SYNERGY_LINK vitamin-c vitamin-e (stv 0.95 0.9))  ; Classic antioxidant pair
(SYNERGY_LINK salicylic-acid niacinamide (stv 0.85 0.8))  ; Acne treatment combo

;; Incompatible combinations (should be avoided or used separately)
(INCOMPATIBILITY_LINK vitamin-c retinol (stv 0.8 0.9))
(INCOMPATIBILITY_LINK retinol salicylic-acid (stv 0.9 0.85))

;; Disputed/cautionary combinations
(INCOMPATIBILITY_LINK vitamin-c niacinamide (stv 0.3 0.6))  ; Mild concern

;; ======================================================================
;; COMPATIBILITY CHECK FUNCTIONS
;; ======================================================================

;; Check if two ingredients are directly compatible
(define compatibility-check
  (BindLink
    (VariableList
      (Variable "$ingredient1")
      (Variable "$ingredient2"))
    (COMPATIBILITY_LINK (Variable "$ingredient1") (Variable "$ingredient2"))
    (EvaluationLink
      (Predicate "compatible")
      (ListLink (Variable "$ingredient1") (Variable "$ingredient2")))))

;; Check for synergistic combinations
(define synergy-check
  (BindLink
    (VariableList
      (Variable "$ingredient1") 
      (Variable "$ingredient2"))
    (SYNERGY_LINK (Variable "$ingredient1") (Variable "$ingredient2"))
    (EvaluationLink
      (Predicate "synergistic")
      (ListLink (Variable "$ingredient1") (Variable "$ingredient2")))))

;; Check for incompatible combinations
(define incompatibility-check
  (BindLink
    (VariableList
      (Variable "$ingredient1")
      (Variable "$ingredient2"))
    (INCOMPATIBILITY_LINK (Variable "$ingredient1") (Variable "$ingredient2"))
    (EvaluationLink
      (Predicate "incompatible")
      (ListLink (Variable "$ingredient1") (Variable "$ingredient2")))))

;; ======================================================================
;; FORMULATION SAFETY ANALYSIS
;; ======================================================================

;; Define a simple formulation
(define test-formulation-1
  (ListLink
    hyaluronic-acid
    niacinamide
    vitamin-e
    glycerin))

(define test-formulation-2
  (ListLink
    vitamin-c
    vitamin-e
    hyaluronic-acid))

(define problematic-formulation
  (ListLink
    vitamin-c
    retinol
    salicylic-acid))

;; ======================================================================
;; ANALYSIS RULES
;; ======================================================================

;; Rule to check all pairs in a formulation for compatibility issues
(define formulation-safety-rule
  (BindLink
    (VariableList
      (Variable "$formulation")
      (Variable "$ingredient1")
      (Variable "$ingredient2"))
    (AndLink
      (MemberLink (Variable "$ingredient1") (Variable "$formulation"))
      (MemberLink (Variable "$ingredient2") (Variable "$formulation"))
      (NotLink (EqualLink (Variable "$ingredient1") (Variable "$ingredient2")))
      (INCOMPATIBILITY_LINK (Variable "$ingredient1") (Variable "$ingredient2")))
    (EvaluationLink
      (Predicate "formulation_has_incompatibility")
      (ListLink 
        (Variable "$formulation") 
        (Variable "$ingredient1") 
        (Variable "$ingredient2")))))

;; Rule to identify beneficial synergies in formulation
(define formulation-synergy-rule
  (BindLink
    (VariableList
      (Variable "$formulation")
      (Variable "$ingredient1")
      (Variable "$ingredient2"))
    (AndLink
      (MemberLink (Variable "$ingredient1") (Variable "$formulation"))
      (MemberLink (Variable "$ingredient2") (Variable "$formulation"))
      (NotLink (EqualLink (Variable "$ingredient1") (Variable "$ingredient2")))
      (SYNERGY_LINK (Variable "$ingredient1") (Variable "$ingredient2")))
    (EvaluationLink
      (Predicate "formulation_has_synergy")
      (ListLink
        (Variable "$formulation")
        (Variable "$ingredient1")
        (Variable "$ingredient2")))))

;; ======================================================================
;; SIMPLE COMPATIBILITY FUNCTIONS
;; ======================================================================

;; Function to get compatibility status between two ingredients
(define (check-pair-compatibility ingredient1 ingredient2)
  (let ((comp-result (cog-execute! 
                       (BindLink
                         (VariableList)
                         (COMPATIBILITY_LINK ingredient1 ingredient2)
                         (ListLink ingredient1 ingredient2))))
        (synergy-result (cog-execute!
                          (BindLink
                            (VariableList)
                            (SYNERGY_LINK ingredient1 ingredient2)
                            (ListLink ingredient1 ingredient2))))
        (incomp-result (cog-execute!
                         (BindLink
                           (VariableList)
                           (INCOMPATIBILITY_LINK ingredient1 ingredient2)
                           (ListLink ingredient1 ingredient2)))))
    (cond
      ((not (equal? synergy-result (SetLink))) "SYNERGISTIC")
      ((not (equal? comp-result (SetLink))) "COMPATIBLE")
      ((not (equal? incomp-result (SetLink))) "INCOMPATIBLE")
      (else "UNKNOWN"))))

;; ======================================================================
;; COMPATIBILITY MATRIX DISPLAY
;; ======================================================================

(define ingredients-list
  (list retinol vitamin-c niacinamide salicylic-acid hyaluronic-acid vitamin-e glycerin))

;; Generate compatibility matrix
(define (generate-compatibility-matrix ingredients)
  (format #t "~%=== INGREDIENT COMPATIBILITY MATRIX ===~%~%")
  (format #t "Legend: S=Synergistic, C=Compatible, I=Incompatible, ?=Unknown~%~%")
  
  ;; Header row
  (format #t "                    ")
  (for-each (lambda (ing)
              (format #t "~8a " (substring (cog-name ing) 0 (min 7 (string-length (cog-name ing))))))
            ingredients)
  (format #t "~%")
  
  ;; Matrix rows
  (for-each
    (lambda (ing1)
      (format #t "~18a " (substring (cog-name ing1) 0 (min 17 (string-length (cog-name ing1)))))
      (for-each
        (lambda (ing2)
          (if (equal? ing1 ing2)
              (format #t "~8a " "---")
              (let ((status (check-pair-compatibility ing1 ing2)))
                (format #t "~8a " 
                  (cond
                    ((string=? status "SYNERGISTIC") "S")
                    ((string=? status "COMPATIBLE") "C") 
                    ((string=? status "INCOMPATIBLE") "I")
                    (else "?"))))))
        ingredients)
      (format #t "~%"))
    ingredients))

;; ======================================================================
;; FORMULATION ANALYSIS FUNCTIONS
;; ======================================================================

(define (analyze-formulation formulation name)
  (format #t "~%=== ANALYZING FORMULATION: ~a ===~%" name)
  (format #t "Ingredients: ")
  (for-each (lambda (ing) 
              (format #t "~a " (cog-name ing))) 
            (cog-outgoing-set formulation))
  (format #t "~%~%")
  
  ;; Check for incompatibilities
  (let ((incompatibilities (cog-execute! 
                             (BindLink
                               (VariableList
                                 (Variable "$ing1")
                                 (Variable "$ing2"))
                               (AndLink
                                 (MemberLink (Variable "$ing1") formulation)
                                 (MemberLink (Variable "$ing2") formulation)
                                 (NotLink (EqualLink (Variable "$ing1") (Variable "$ing2")))
                                 (INCOMPATIBILITY_LINK (Variable "$ing1") (Variable "$ing2")))
                               (ListLink (Variable "$ing1") (Variable "$ing2"))))))
    
    (if (equal? incompatibilities (SetLink))
        (format #t "✓ No incompatibilities detected~%")
        (begin
          (format #t "⚠ INCOMPATIBILITIES DETECTED:~%")
          (for-each
            (lambda (pair)
              (let ((ing1 (gar pair))
                    (ing2 (gdr pair)))
                (format #t "  - ~a + ~a~%" (cog-name ing1) (cog-name ing2))))
            (cog-outgoing-set incompatibilities)))))
  
  ;; Check for synergies
  (let ((synergies (cog-execute!
                     (BindLink
                       (VariableList
                         (Variable "$ing1")
                         (Variable "$ing2"))
                       (AndLink
                         (MemberLink (Variable "$ing1") formulation)
                         (MemberLink (Variable "$ing2") formulation)
                         (NotLink (EqualLink (Variable "$ing1") (Variable "$ing2")))
                         (SYNERGY_LINK (Variable "$ing1") (Variable "$ing2")))
                       (ListLink (Variable "$ing1") (Variable "$ing2"))))))
    
    (if (equal? synergies (SetLink))
        (format #t "○ No synergies detected~%")
        (begin
          (format #t "✓ SYNERGIES DETECTED:~%")
          (for-each
            (lambda (pair)
              (let ((ing1 (gar pair))
                    (ing2 (gdr pair)))
                (format #t "  + ~a + ~a~%" (cog-name ing1) (cog-name ing2))))
            (cog-outgoing-set synergies))))))

;; ======================================================================
;; EXECUTION AND RESULTS
;; ======================================================================

;; Display compatibility matrix
(generate-compatibility-matrix ingredients-list)

;; Analyze test formulations
(analyze-formulation test-formulation-1 "GENTLE HYDRATING SERUM")
(analyze-formulation test-formulation-2 "ANTIOXIDANT VITAMIN C SERUM")  
(analyze-formulation problematic-formulation "PROBLEMATIC MULTI-ACTIVE SERUM")

;; ======================================================================
;; INDIVIDUAL INGREDIENT PAIR CHECKS
;; ======================================================================

(format #t "~%=== INDIVIDUAL COMPATIBILITY CHECKS ===~%")

(define test-pairs
  (list
    (list vitamin-c vitamin-e "Vitamin C + Vitamin E")
    (list vitamin-c retinol "Vitamin C + Retinol")
    (list retinol salicylic-acid "Retinol + Salicylic Acid")
    (list hyaluronic-acid niacinamide "Hyaluronic Acid + Niacinamide")
    (list salicylic-acid niacinamide "Salicylic Acid + Niacinamide")
    (list vitamin-c niacinamide "Vitamin C + Niacinamide")))

(for-each
  (lambda (pair-info)
    (let ((ing1 (car pair-info))
          (ing2 (cadr pair-info))
          (name (caddr pair-info)))
      (let ((status (check-pair-compatibility ing1 ing2)))
        (format #t "~a: ~a~%" name status))))
  test-pairs)

;; ======================================================================
;; RECOMMENDATIONS
;; ======================================================================

(format #t "~%=== FORMULATION RECOMMENDATIONS ===~%")
(format #t "Based on compatibility analysis:~%~%")

(format #t "SAFE COMBINATIONS:~%")
(format #t "• Hyaluronic Acid + Niacinamide: Excellent hydration and pore control~%")
(format #t "• Vitamin C + Vitamin E: Synergistic antioxidant protection~%")
(format #t "• Salicylic Acid + Niacinamide: Effective acne treatment~%")
(format #t "• Hyaluronic Acid + Glycerin: Enhanced moisturization~%")

(format #t "~%AVOID COMBINING:~%")
(format #t "• Vitamin C + Retinol: pH conflict and potential irritation~%")
(format #t "• Retinol + Salicylic Acid: Risk of over-exfoliation~%")

(format #t "~%USE WITH CAUTION:~%")
(format #t "• Vitamin C + Niacinamide: Disputed interaction, monitor pH~%")

(format #t "~%USAGE TIPS:~%")
(format #t "• Separate incompatible actives by time (AM/PM) or days~%")
(format #t "• Start with lower concentrations when combining actives~%")
(format #t "• Always patch test new combinations~%")
(format #t "• Consider pH requirements for optimal stability~%")

(format #t "~%=== COMPATIBILITY ANALYSIS COMPLETE ===~%")
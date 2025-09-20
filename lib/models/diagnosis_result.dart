class DiagnosisResult {
  final String predictedClass;
  final String className;
  final double confidence;
  final Map<String, double> allProbabilities;

  DiagnosisResult({
    required this.predictedClass,
    required this.className,
    required this.confidence,
    required this.allProbabilities,
  });

  factory DiagnosisResult.fromJson(Map<String, dynamic> json) {
    return DiagnosisResult(
      predictedClass: json['predicted_class'] ?? '',
      className: json['class_name'] ?? '',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      allProbabilities: Map<String, double>.from(
        json['all_probabilities'] ?? {},
      ),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'predicted_class': predictedClass,
      'class_name': className,
      'confidence': confidence,
      'all_probabilities': allProbabilities,
    };
  }

  @override
  String toString() {
    return 'DiagnosisResult(predictedClass: $predictedClass, className: $className, confidence: $confidence)';
  }
}
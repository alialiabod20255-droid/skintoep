class User {
  final int? id;
  final String name;
  final String email;
  final String role;
  final String? profileImage;
  final String? phone;
  final String? specialization;
  final String? licenseNumber;
  final DateTime createdAt;
  final DateTime? updatedAt;

  User({
    this.id,
    required this.name,
    required this.email,
    required this.role,
    this.profileImage,
    this.phone,
    this.specialization,
    this.licenseNumber,
    required this.createdAt,
    this.updatedAt,
  });

  User copyWith({
    int? id,
    String? name,
    String? email,
    String? role,
    String? profileImage,
    String? phone,
    String? specialization,
    String? licenseNumber,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return User(
      id: id ?? this.id,
      name: name ?? this.name,
      email: email ?? this.email,
      role: role ?? this.role,
      profileImage: profileImage ?? this.profileImage,
      phone: phone ?? this.phone,
      specialization: specialization ?? this.specialization,
      licenseNumber: licenseNumber ?? this.licenseNumber,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'email': email,
      'role': role,
      'profileImage': profileImage,
      'phone': phone,
      'specialization': specialization,
      'licenseNumber': licenseNumber,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt?.toIso8601String(),
    };
  }

  factory User.fromMap(Map<String, dynamic> map) {
    return User(
      id: map['id']?.toInt(),
      name: map['name'] ?? '',
      email: map['email'] ?? '',
      role: map['role'] ?? '',
      profileImage: map['profileImage'],
      phone: map['phone'],
      specialization: map['specialization'],
      licenseNumber: map['licenseNumber'],
      createdAt: DateTime.parse(map['createdAt']),
      updatedAt: map['updatedAt'] != null ? DateTime.parse(map['updatedAt']) : null,
    );
  }

  String get displayRole {
    switch (role) {
      case 'doctor':
        return 'طبيب';
      case 'nurse':
        return 'ممرض/ممرضة';
      case 'admin':
        return 'مدير';
      default:
        return role;
    }
  }

  String get initials {
    final names = name.split(' ');
    if (names.length >= 2) {
      return '${names[0][0]}${names[1][0]}';
    } else if (names.isNotEmpty) {
      return names[0][0];
    }
    return 'U';
  }

  @override
  String toString() {
    return 'User(id: $id, name: $name, email: $email, role: $role)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is User && other.id == id;
  }

  @override
  int get hashCode => id.hashCode;
}
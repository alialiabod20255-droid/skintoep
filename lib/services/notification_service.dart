import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:permission_handler/permission_handler.dart';

class NotificationService {
  static final FlutterLocalNotificationsPlugin _notifications =
      FlutterLocalNotificationsPlugin();

  static Future<void> initialize() async {
    // طلب الإذن للإشعارات
    await Permission.notification.request();

    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosSettings = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );

    const initSettings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _notifications.initialize(
      initSettings,
      onDidReceiveNotificationResponse: _onNotificationTapped,
    );
  }

  static void _onNotificationTapped(NotificationResponse response) {
    // التعامل مع النقر على الإشعار
    print('Notification tapped: ${response.payload}');
  }

  static Future<void> showDiagnosisComplete({
    required String patientName,
    required String result,
    required double confidence,
  }) async {
    const androidDetails = AndroidNotificationDetails(
      'diagnosis_channel',
      'تشخيص الأمراض',
      channelDescription: 'إشعارات اكتمال التشخيص',
      importance: Importance.high,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    const details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notifications.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      'اكتمل التشخيص',
      'تم تشخيص $patientName: $result (${(confidence * 100).toInt()}%)',
      details,
      payload: 'diagnosis_complete',
    );
  }

  static Future<void> showAppointmentReminder({
    required String patientName,
    required DateTime appointmentTime,
  }) async {
    const androidDetails = AndroidNotificationDetails(
      'appointment_channel',
      'تذكير المواعيد',
      channelDescription: 'تذكير بمواعيد المرضى',
      importance: Importance.high,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    const details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notifications.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      'تذكير موعد',
      'موعد مع $patientName في ${_formatTime(appointmentTime)}',
      details,
      payload: 'appointment_reminder',
    );
  }

  static Future<void> scheduleAppointmentReminder({
    required int id,
    required String patientName,
    required DateTime appointmentTime,
  }) async {
    const androidDetails = AndroidNotificationDetails(
      'appointment_channel',
      'تذكير المواعيد',
      channelDescription: 'تذكير بمواعيد المرضى',
      importance: Importance.high,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    const details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    // جدولة الإشعار قبل 15 دقيقة من الموعد
    final reminderTime = appointmentTime.subtract(const Duration(minutes: 15));

    await _notifications.schedule(
      id,
      'تذكير موعد',
      'موعد مع $patientName خلال 15 دقيقة',
      reminderTime,
      details,
      payload: 'scheduled_appointment',
    );
  }

  static Future<void> showSystemNotification({
    required String title,
    required String body,
    String? payload,
  }) async {
    const androidDetails = AndroidNotificationDetails(
      'system_channel',
      'إشعارات النظام',
      channelDescription: 'إشعارات عامة من النظام',
      importance: Importance.defaultImportance,
      priority: Priority.defaultPriority,
      icon: '@mipmap/ic_launcher',
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    const details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notifications.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      details,
      payload: payload,
    );
  }

  static Future<void> cancelNotification(int id) async {
    await _notifications.cancel(id);
  }

  static Future<void> cancelAllNotifications() async {
    await _notifications.cancelAll();
  }

  static String _formatTime(DateTime dateTime) {
    final hour = dateTime.hour.toString().padLeft(2, '0');
    final minute = dateTime.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }
}
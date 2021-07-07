import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:multilevel_diacritizer_ui/main.dart';

void main() {

  final app = MultilevelDiacritizerFrontEnd();

  testWidgets('DiacritizationForm look', (WidgetTester tester) async {
    await tester.pumpWidget(app);
    expect(find.text('Multilevel diacritizer'), findsOneWidget);
    expect(find.byType(SnackBar), findsNothing);
    expect(find.byType(AppBar), findsOneWidget);
    expect(find.byType(TextFormField), findsOneWidget);
    expect(find.text('Restore diacritics'), findsOneWidget);
  });

  testWidgets('DiacritizationForm text interaction', (WidgetTester tester) async {
    await tester.pumpWidget(app);
    final text = 'مرحبا!';
    await tester.enterText(find.byType(TextFormField), text);
    expect(find.widgetWithText(TextFormField, text), findsOneWidget);
  });

  testWidgets('DiacritizationForm side buttons', (WidgetTester tester) async {
    await tester.pumpWidget(app);
    final copyButton = find.byIcon(Icons.copy);
    final pastButton = find.byIcon(Icons.paste);
    final clearButton = find.byIcon(Icons.clear);
    expect(copyButton, findsOneWidget);
    expect(pastButton, findsOneWidget);
    expect(clearButton, findsOneWidget);
    final text = 'نص قصير';
    await tester.enterText(find.byType(TextFormField), text);
    await tester.tap(clearButton);
    await tester.pump();
    expect(find.byWidgetPredicate((widget) => widget is TextFormField ? widget.controller?.text == '' : false), findsOneWidget);
  });

  testWidgets('DiacritizationForm Restore diacritics button', (WidgetTester tester) async {
    await tester.pumpWidget(app);
    final restoreButton = find.text('Restore diacritics');
    await tester.tap(restoreButton);
    await tester.pump();
    expect(find.byType(SnackBar), findsOneWidget);
  });

}

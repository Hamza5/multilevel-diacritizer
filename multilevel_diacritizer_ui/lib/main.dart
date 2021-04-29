import 'dart:convert';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:clipboard/clipboard.dart';

import 'dart:html' as html;

final Uri _submitURL = Uri.parse(html.window.location.href);

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Multilevel diacritizer',
      builder: (BuildContext context, Widget? child) => FocusScope(
        child: GestureDetector(
          onTap: () => Focus.of(context).unfocus(),
          child: child,
        ),
      ),
      theme: ThemeData(
          primarySwatch: Colors.blue,
          visualDensity: VisualDensity.adaptivePlatformDensity,
          accentColor: Colors.green,
          iconTheme: IconTheme.of(context).copyWith(color: Colors.blue)),
      home: Scaffold(
        appBar: AppBar(
          title: Text('Multilevel diacritizer'),
        ),
        body: DiacritizationForm(),
      ),
    );
  }
}

class DiacritizationForm extends StatefulWidget {
  @override
  _DiacritizationFormState createState() => _DiacritizationFormState();
}

class _DiacritizationFormState extends State<DiacritizationForm> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();
  final TextEditingController _textEditingController = TextEditingController();
  bool _enableSubmit = true;

  Future<void> submit() async {
    setState(() {
      _enableSubmit = false;
    });
    _showSnackBar(
        'Diacritizing the text...', Icons.text_fields, Colors.blueAccent,
        duration: Duration(minutes: 3));
    var finalMessage = 'Text diacritized!';
    var finalIcon = Icons.check;
    var finalColor = Theme.of(context).accentColor;
    try {
      var response = await http.post(_submitURL,
          body: _textEditingController.text,
          headers: {'Content-Type': 'text/plain; charset=UTF-8'},
          encoding: Encoding.getByName('UTF-8'));
      setState(() {
        _textEditingController.text = response.body;
      });
    } catch (e) {
      finalMessage = e.toString();
      finalIcon = Icons.warning;
      finalColor = Theme.of(context).errorColor;
    } finally {
      setState(() {
        _enableSubmit = true;
      });
      _showSnackBar(finalMessage, finalIcon, finalColor);
    }
  }

  void _showSnackBar(String text, IconData iconData, Color background,
      {Duration duration = const Duration(seconds: 4)}) {
    var snackBar = SnackBar(
      content: ListTile(
        leading: Icon(
          iconData,
          color: Theme.of(context).scaffoldBackgroundColor,
        ),
        title: Text(
          text,
          style: TextStyle(color: Theme.of(context).scaffoldBackgroundColor),
        ),
      ),
      backgroundColor: background,
      duration: duration,
    );
    ScaffoldMessenger.of(context).removeCurrentSnackBar();
    ScaffoldMessenger.of(context).showSnackBar(snackBar);
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: Container(
        padding: EdgeInsets.all(15),
        child: Column(
          children: [
            Row(
              children: <Widget>[
                Expanded(
                  child: TextFormField(
                    readOnly: !_enableSubmit,
                    textDirection: TextDirection.rtl,
                    style: TextStyle(
                      fontSize: Theme.of(context).textTheme.headline6?.fontSize,
                    ),
                    minLines: 10,
                    maxLines: 100,
                    decoration: InputDecoration(
                      labelText: 'The Arabic text to diacritize',
                      labelStyle: Theme.of(context)
                          .textTheme
                          .headline6
                          ?.copyWith(color: Colors.blue),
                      border: OutlineInputBorder(),
                    ),
                    controller: _textEditingController,
                    autofocus: true,
                  ),
                ),
                Column(
                  children: <Widget>[
                    IconButton(
                      icon: Icon(Icons.copy),
                      onPressed: () =>
                          FlutterClipboard.copy(_textEditingController.text)
                              .then((value) => _showSnackBar(
                                  'Text copied to clipboard!',
                                  Icons.info,
                                  Colors.blueAccent)),
                      tooltip: 'Copy the text to the clipboard',
                    ),
                    SizedBox(
                      height: 10,
                    ),
                    IconButton(
                      icon: Icon(Icons.paste),
                      onPressed: () => FlutterClipboard.paste().then((value) {
                        setState(() => _textEditingController.text = value);
                      }),
                      tooltip: 'Paste the text from the clipboard',
                    ),
                    SizedBox(
                      height: 10,
                    ),
                    IconButton(
                        icon: Icon(Icons.clear),
                        onPressed: () {
                          setState(() => _textEditingController.clear());
                        },
                      tooltip: 'Clear the text',
                    )
                  ],
                ),
              ],
            ),
            SizedBox(
              height: 5,
            ),
            ElevatedButton.icon(
              onPressed: _enableSubmit ? submit : null,
              label: Text(
                'Restore diacritics',
                textScaleFactor: 1.5,
              ),
              icon: Icon(
                Icons.settings_backup_restore,
                color: Theme.of(context).scaffoldBackgroundColor,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

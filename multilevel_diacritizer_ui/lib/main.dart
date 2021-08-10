import 'dart:convert';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:clipboard/clipboard.dart';


void main() {
  runApp(MultilevelDiacritizerFrontEnd());
}

class MultilevelDiacritizerFrontEnd extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Multilevel diacritizer',
      builder: (BuildContext context, Widget? child) => FocusScope(
        child: GestureDetector(
          // Useful to hide the keyboard in the mobile app
          onTap: () => Focus.of(context).unfocus(),
          child: child,
        ),
      ),
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        accentColor: Colors.green,
        iconTheme: IconTheme.of(context).copyWith(color: Colors.blue),
      ),
      home: Scaffold(
        appBar: AppBar(
          title: Text('Multilevel diacritizer'),
        ),
        body: Container(
          padding: const EdgeInsets.only(top: 20, left: 20, right: 20),
          child: ListView(
              children: [
                DiacritizationForm(),
                const SizedBox(height: 20),
                const InformationArea(),
                const SizedBox(height: 20,)
              ]
          ),
        ),
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
  double _angle = 0;

  Future<void> submit() async {
    setState(() {
      _enableSubmit = false;
      _angle -= pi;
    });
    _showSnackBar(
        'Diacritizing the text...', Icons.text_fields, Colors.blueAccent,
        duration: Duration(minutes: 3));
    var finalMessage = 'Text diacritized!';
    var finalIcon = Icons.check;
    var finalColor = Theme.of(context).accentColor;
    try {
      var response = await http.post(Uri.parse(''),
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
      child: Column(
        children: [
          Row(
            children: <Widget>[
              Expanded(
                child: Stack(
                  children: [
                    TextFormField(
                      readOnly: !_enableSubmit,
                      enabled: _enableSubmit,
                      textDirection: TextDirection.rtl,
                      style: TextStyle(
                        fontSize:
                            Theme.of(context).textTheme.headline6?.fontSize,
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
                    Container(
                      padding: EdgeInsets.all(1),
                      child: Visibility(
                        visible: !_enableSubmit,
                        child: ClipRRect(
                          child: LinearProgressIndicator(
                            minHeight: 5,
                            valueColor:
                                AlwaysStoppedAnimation(Colors.blueAccent),
                          ),
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    ),
                  ],
                  alignment: Alignment.bottomCenter,
                ),
              ),
              Column(
                children: <Widget>[
                  IconButton(
                    icon: Icon(Icons.copy),
                    onPressed: _enableSubmit
                        ? () =>
                            FlutterClipboard.copy(_textEditingController.text)
                                .then(
                                    (value) =>
                                        _showSnackBar(
                                            'Text copied to clipboard!',
                                            Icons.info,
                                            Colors.blueAccent))
                                .catchError((error) => _showSnackBar(
                                    error.toString(),
                                    Icons.warning,
                                    Theme.of(context).errorColor))
                        : null,
                    tooltip: 'Copy the text to the clipboard',
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  IconButton(
                    icon: Icon(Icons.paste),
                    onPressed: _enableSubmit
                        ? () => FlutterClipboard.paste().catchError((error) {
                              _showSnackBar(error.toString(), Icons.warning,
                                  Theme.of(context).errorColor);
                              return '';
                            }).then((value) {
                              setState(
                                  () => _textEditingController.text = value);
                            })
                        : null,
                    tooltip: 'Paste the text from the clipboard',
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  IconButton(
                    icon: Icon(Icons.clear),
                    onPressed: _enableSubmit
                        ? () {
                            setState(() => _textEditingController.clear());
                          }
                        : null,
                    tooltip: 'Clear the text',
                  )
                ],
              ),
            ],
          ),
          SizedBox(
            height: 5,
          ),
          FittedBox(
            fit: BoxFit.scaleDown,
            child: ElevatedButton.icon(
              onPressed: _enableSubmit ? submit : null,
              label: Text(
                'Restore diacritics',
                textScaleFactor: 1.5,
              ),
              icon: TweenAnimationBuilder(
                child: Icon(
                  Icons.settings_backup_restore,
                  color: Theme.of(context).scaffoldBackgroundColor,
                ),
                duration: Duration(milliseconds: 500),
                tween: Tween<double>(begin: 0, end: _angle),
                builder: (BuildContext context, double angle, Widget? child) {
                  return Transform.rotate(
                    angle: angle,
                    child: child,
                  );
                },
                onEnd: () => setState(() {
                  if (!_enableSubmit) _angle -= pi;
                }),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class InformationArea extends StatelessWidget {
  const InformationArea({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      shadowColor: Theme.of(context).primaryColorDark,
      elevation: 2,
      color: Theme.of(context).primaryColorLight,
      child: DefaultTextStyle(
        style: TextStyle(
          color: Theme.of(context).primaryColorDark,
          fontSize: 16,
        ),
        child: Column(
          children: [
            Card(
              color: Theme.of(context).primaryColor,
              margin: const EdgeInsets.only(bottom: 5),
              child: Padding(
                padding: EdgeInsets.all(5),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.info,
                      color: Theme.of(context).scaffoldBackgroundColor,
                    ),
                    SizedBox(width: 10,),
                    Text(
                      'Information',
                      style: TextStyle(
                        color: Theme.of(context).scaffoldBackgroundColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                      ),
                    ),
                  ]
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(10),
              child: Text.rich(
                TextSpan(
                  text: 'This web application allow the user to predict the'
                      ' diacritics of a diacritics-free Arabic text. The text '
                      'can be simply inserted in the text field above then sent'
                      ' to the server by clicking the ',
                  children: [
                    TextSpan(
                      text: 'Restore diacritcs',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    TextSpan(
                      text: ' button. When the server response arrive, the '
                          'diacritized text will be used to overwrite the '
                          'original text.',
                    )
                  ]
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(left: 10, right: 10, bottom: 10),
              child: Text.rich(
                TextSpan(
                  text: 'You can use this application as an API by just sending'
                      ' an HTTP POST request to the same URL that this '
                      'application is served from with the Arabic text as the '
                      'body (',
                  children: [
                    TextSpan(
                      text: 'Content-Type: text/plain',
                      style: TextStyle(
                        backgroundColor: Theme.of(context).canvasColor,
                      ),
                    ),
                    TextSpan(
                      text: '). The response will contain the diacritized text.'
                    )
                  ]
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

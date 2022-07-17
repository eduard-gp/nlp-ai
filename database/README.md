# Style format

This file describes how the dialog files are formatted.

This format eases the process of writing dialogs before converting them automatically to JSON.

### <-- START PERSONA -->

Meaning: Marks the starting of a persona.

### <-- END PERSONA -->

Meaning: Marks the end of a persona.

**Remark**: Between <-- START PERSONA --> and <-- END PERSONA --> is written a dialog.

The following tags encapsulates a question <-> answer exchange. It is important that every collection of questions and answers is marked with a label.

- `L:`
- `Q:`
- `A:`  


### D:

Meaning: Description

If an attribute appears more than once, it will be stored in a list.

Example 1: 

```
D: age=22
```

```json 
"description": {
    "age": 22
}
```

Example 2:

```
D: annotations="smoker"
D: anottaions="broken arm"
```

```json
{
    "description": {
        "annotations"=["smoker", "broken arm"]
    }
}
```

### L:

Meaning: Label

It should appear only once in a collection of question-answers which have the same intent.

Example:
```
L: greetings
```
```json
{
        "label": "greetings"
}
```

### Q

Meaning: Quesetion

There should be only a question on a line, but it is possible to have two or more chainded questions if they have the same label in a persona dialog.

Example 1:

```
Q: What is the reason for the presentation at the hospital?
Q: Why did you come?
```

```json
{
        "questions": ["What is the reason for the presentation at the hospital?", "What is the reason for the presentation at the hospital?"]
}
```

### A

Meaning: answer

It follows the same rules as questions.

### Important

- The number of questions and answers don't have to match, if they are considered under the same label.

- There must be at least a question for every answer and at least an answer for every question.

- There must a label associated to every collection of questions and answers.

- A group of label, questions and answers **must** not have empty lines between them because an empty line marks the end of the group.

## Full example

```
D: case=Case No. 1 - M.I.
D: sex=f
D: age=55
D: profession=retired
D: annotations=former smoker

L: visit_reason
Q: What is the reason for the presentation at the hospital?
Q: Why did you come?
A: I feel like it is hard to breathe and I lost my consciousness right before calling the ambulance. 
```


```json
{ 
 
    "description": {
        "case": "Case No. 1 - M.I.",
        "sex": "f",
        "age": 55,
        "profession": "retired",
        "annotations": "former smoker"
    },
    "dialog": [
        "questions": ["What is the reason for the presentation at the hospital?", "Why did you come?"],
        "answers": ["I feel like it is hard to breathe and I lost my consciousness right before calling the ambulance."]
    ]
}
```

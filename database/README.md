# Style format

This file describes how the dialog files are formatted.

## <-- START PERSONA -->

Meaning: Marks the starting of a persona.

## <-- END PERSONA -->

Meaning: Marks the end of a persona.

**Remark**: Between <-- START PERSONA --> and <-- END PERSONA --> is written a dialog.

The following tags encapsulates a question <-> answear exchange. It is important that every collection of questions and answears is marked with a label.

- `L`
- `Q`
- `A`  


## D:

Meaning: Description

Usage: D: attribute=value

If an attribute appears more than once, it will be stored in a list.

Example: 

1. > D: age=22
    
    "description": {
        "age": 22
    }

2. > D: annotations="smoker"
   >
   > D: anottaions="broken arm"

   `{
        "description": {
            "annotations"=["smoker", "broken arm"]
        }
   }`

## L:

Meaning: Label

It should appear only once in a collection of question-answears which have the same intent.

Example:

1. > L: greetings

    `{
        "label": "greetings"
    }`

## Q

Meaning: Quesetion

There should be only a question on a line, but it is possible to have two or more chainded questions if they have the same label in a persona dialog.

Example:

1.  > Q: What is the reason for the presentation at the hospital?
    >
    > Q: Why did you come?

    `{
        "questions": ["What is the reason for the presentation at the hospital?", "What is the reason for the presentation at the hospital?"]
    }`

## A

Meaning: Answear

It follows the same rules as questions.

## Important

The number of questions and answears don't have to match, but they are under the same label.

There must be at least a question for every answear and at least an answear for every question.

There must a label associated to every collection of questions and answears.

A group of label, questions and answears **must** not have empty lines between them because an empty line marks the end of the group.

## Full example

> D: case=Case No. 1 - M.I.
>
> D: sex=f
>
> D: age=55
>
> D: profession=retired
>
> D: annotations=former smoker
>
> L: visit_reason
>
> Q: What is the reason for the presentation at the hospital?
>
> Q: Why did you come?
>
> A: I feel like it is hard to breathe and I lost my consciousness right before calling the ambulance. 

`{ 
 
    description: {
        "case: "Case No. 1 - M.I.",
        "sex: "f",
        "age": 55,
        "profession": "retired",
        "annotations": "former smoker"
    },
    dialog: [
        "questions": ["What is the reason for the presentation at the hospital?", "Why did you come?"],
        "answears": ["I feel like it is hard to breathe and I lost my consciousness right before calling the ambulance."]
    ]
}`

# The measurement of the capture

The capture of the screen is measured on a set of videos. Every video
answers a different question, and every one of them was checked to see
whether the text on the screen can be read at all. A video where the
text cannot be read says nothing about the capture, it only says that
the reading of the text is not the problem.

The set is in `manifest.json`. The videos are not in the repository,
they are too big for it, and `data_root` in the manifest says where they
are.

## The kinds of video

| Kind | What it is | What it measures |
| :--- | :--- | :--- |
| `A_slides` | Printed slides that fill the frame | The reading of the text, and whether the moments are picked |
| `C_code` | An editor with a picture in it | Text that changes line by line, not slide by slide |
| `D_document` | A paper in two columns | Small text in the middle of the screen |
| `E_talking_head` | A person talking | That **nothing** is captured |
| `F_messy` | A browser with a menu bar and tabs in it | That the things that are not the lecture are left out |
| `G_hinglish` | A lecture in Hindi and English together | The case the notes have to work in |
| `H_handwriting` | A board written on by hand | That the reading of printed text cannot read it, and that this is known |

## How it is run

```bash
# the text of the screen, every two seconds, of every video
python eval/extract_readings.py --all --step 2 --workers 8

# what the moments of the change are, against the notes
python eval/score.py --video A_mit_6_0001_spring2018_lecture1 --trigger

# what a system read, against the notes
python eval/score.py --video A_mit_6_0001_spring2018_lecture1 --captures captures.json
```

The readings take about two hours for the whole set. The work is split
over eight hands, and each one is given a single thread, because the
runtime uses every core of the machine by itself and they fight over
them when they run side by side.

## The numbers

- **The reading**: how many of the words of the slide were found, in
  both directions. The share of the words is used and not the character
  error rate, because a slide shows only half of its text while it is
  being animated and the character error rate punishes that heavily.
- **The moments**: how many of the changes of the screen were picked, and
  how many of the picks were not a change at all. A picture that fits no
  page of the notes is not a slide, and picking it is a mistake.
- **The cost**: how many pictures were looked at, and how many of them
  were sent to a model.

## The reading of the screen

`RapidOCR` reads the text. It runs on the processor, without a model
that has to be asked, and it is the same reading that the capture will
use. A picture of 1920x1080 takes between one and nine seconds to
read, so the capture must not read every picture. The cheap difference
of the pictures says which ones are worth reading.

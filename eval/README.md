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

## What was measured

The numbers below are from the lecture of MIT 6.0001 in a 1080p picture,
read every two seconds, 1905 pictures over 64 minutes. The ground truth
is the sequence of slides of the course, which comes from the notes of
the course and says nothing about how the screen is watched.

| The trigger | Fires | Slides caught | Slides missed |
| :--- | ---: | ---: | ---: |
| The gate of the pixels, at a change of 0.4% | 317 | 25 of 25 | 0 |
| The gate of the pixels, at a change of 2% | 297 | 25 of 25 | 0 |
| The gate of the pixels, at a change of 10% | 251 | 20 of 25 | 5 |
| The new text of the screen, 8 new words | 92 | 17 of 25 | 8 |

**The gate of the pixels is the better one.** It does not miss a single
slide. The text of the screen is not as good a way to know that the
screen changed, because two slides of the same lecture share a heading,
and the reading of the screen is not the same in every picture.

**But the gate of the pixels fires 297 times in a lecture.** At five
requests a minute that is five hours of requests, and every one of them
carries a whole picture of the screen.

So the text of the screen is not used to decide **whether** to look at
the screen. It is used to decide **what to send** and **what to leave
for the model**. The text is read on the machine, where it is free,
and the model is only asked for the parts that the reading cannot do:
a diagram, a formula, and the meaning of a slide.

| | Before | After |
| :--- | ---: | ---: |
| The requests to the model in a lecture | 297 | 0 to 5 |
| The tokens of a request | 1000 to 40000 | 200 to 800 |
| The tokens of a lecture | 9.3 million | 40000 |
| The reading of the text of the screen | 0.61 F1 | 0.61 F1 |
| The slides missed | 0 | 0 |

The reading of the screen was measured against the notes of the course:
the share of the words of the slide that were read, in both directions.
The model reads the same text just as well, at three hundred times the
price.

## The numbers that are measured


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

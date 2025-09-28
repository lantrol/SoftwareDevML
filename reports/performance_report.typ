= Practice 2 Performance Report
== Indications
Create a minimal report (1 or 2 pages) that includes your key performance
visualizations with brief interpretations

- Confusion matrices (if applicable).
- Model calibration curve (see the last section of this document).
- Samples with highest loss values.
- Training losses, accuracies, etc.

Ensure proper figure formatting with axis labels, titles, and legends. Save figures at reasonable
resolutions. Structure your code to be modular and reusable.
Your report should be in PDF or HTML form, but include the original file too (word, markdown,
typst…). Include figure captions and brief explanations of what each visualization reveals.

== Data Analysis
Before we start, we will analize the data distribution of our dataset, both the splits
distribution and per-class distributions.

#figure(
  grid(
    rows: 2,
    image("figures/class_distribution_facets.png", width: 100%),
    image("figures/class_distribution.png", width: 80%),
  ),
  caption: [Class distribution between dataset splits]
)

The dataset is *balanced*. Each split has the same ratio of *smoking* and *non-smoking* images
in it. The dataset is divided in 80% for train and validation, and the rest 20% for test.

To each image of the dataset, we've manually added new data to it: *genre* and *category*:
- Genres: Man and Woman
- Categories: Random, Phone, Inhalador, Water and Cough

With this new information, we can visualize the distribution of the image set.

#figure(
  image("figures/genre_distribution.png", width: 100%),
  caption: [Genre distribution between categories and classes]
)

We can see the ratio between *Man* and *Woman* in the image set is not balanced.
There is a bigger women ration in the *Smoking* image class in comparison to *Man*.
Between the image categories, there is a higher amount of Man images without a clear
action, while there are more Woman images of drinking water. Other categories are balanced.

#figure(
  grid(
    columns: 2,
    image("figures/wrong_pred_per_genre.png", width: 100%),
    image("figures/wrong_pred_per_category.png", width: 100%),
  ),
  caption: [Class distribution between dataset splits]
)


== Confusion matrix

== Training losses & accuracies

== Model calibration curve

== Samples with highest loss values

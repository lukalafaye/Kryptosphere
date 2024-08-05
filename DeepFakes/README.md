# DeepFakes

Les deepfakes sont des vidéos ou des images créées à l'aide de l'intelligence artificielle et de l'apprentissage automatique pour superposer le visage ou la voix d'une personne sur une autre. En d'autres termes, les deepfakes permettent de créer des vidéos ou des images qui donnent l'impression qu'une personne dit ou fait quelque chose qu'elle n'a jamais dit ou fait.

Les deepfakes peuvent être utilisés à des fins malveillantes, telles que la diffusion de fausses informations ou la diffamation de personnes en faisant circuler des vidéos ou des images trompeuses. Ils peuvent également être utilisés à des fins de divertissement ou de création artistique.

Il est important de noter que les deepfakes peuvent être très convaincants et difficiles à détecter, ce qui soulève des préoccupations quant à leur potentiel d'abus et de manipulation. C'est pourquoi il est crucial de sensibiliser le public à la présence de deepfakes et de développer des outils pour les détecter et les signaler.

## Vocabulaire anglais

### 1. First order vs second order information

-> Human faces are hierarchical visual stimuli in the sense that they convey at least two levels of information. The most basic attributes that are repeated in every face (i.e., two eyes, above a nose, above a mouth) provide “first-order information” and this can be used to distinguish faces from other visual objects (face detection).  

Because all faces share the same first-order configuration, the identification of an individual face requires information about the ways that one face differs from any other.
-> Second-order information” refers to the variance that exists between faces, such as the distance between the eyes

### 2. Face inversion discovery

The face inversion effect is a phenomenon where identifying inverted (upside-down) faces compared to upright faces is much more difficult than doing the same for non-facial objects.

The most supported explanation for why faces take longer to recognise when they are inverted is the configural information hypothesis. 
* The configural information hypothesis states that faces are processed with the use of configural information to form a holistic (whole) representation of a face. 
* Objects, however, are not processed in this configural way. Instead, they are processed featurally (in parts). 
Each part of the object is processed independently to allow it to be recognised. This is known as a featural recognition method.
Inverting a face disrupts configural processing, forcing it to instead be processed featurally like other objects. This causes a delay since it takes longer to form a representation of a face with only local information.

### 3. Configural information hypothesis

According to the configural information hypothesis, the face inversion effect occurs because configural information can no longer be used to build a holistic representation of a face. Inverted faces are instead processed like objects, using local information (i.e. the individual features of the face) instead of configural information. 

It has been suggested that faces and objects are both recognised using featural processing mechanisms, instead of holistic processing for faces and featural processing for objects.
The face inversion effect is not caused by delay from faces being processed as objects. Instead, another element is involved. Two potential explanations follow. 

#### Perceptual learning

According to the perceptual learning theory, being presented with a stimulus (for example, faces or cars) more often makes that stimulus easier to recognise in the future.
Most people are highly familiar with viewing upright faces. It follows that highly efficient mechanisms have been able to develop to the quick detection and identification of upright faces.
This means that the face inversion effect would therefore be caused by an increased amount of experience with perceiving and recognising upright faces compared to inverted faces.

### 4. Face-scheme incompatibility

Proposed to explain more of the missing elements of the configurational information hypothesis
The model defines a scheme as an abstract representation of the general structure of a face, including characteristics common to most faces (i.e. the structure of and relationships between facial features).
A prototype refers to an image of what an average face would look like for a particular group (e.g. humans or monkeys). After being recognised as a face with the use of a scheme, new faces are added to a group by being evaluated for their similarity to that group's prototype.

There are different schemes for upright and inverted faces: upright faces are more frequently viewed and thus have more efficient schemes than inverted faces. The face inversion effect is thus partly caused by less efficient schemes for processing the less familiar inverted form of faces. This makes the face-scheme incompatibility model similar to the perceptual learning theory, because both consider the role of experience important in the quick recognition of faces.

### 5. Creating motion

Sparse techniques only need to process some pixels from the whole image.
Dense techniques process all the pixels. Dense motion = 1 vector per pixel

## Slides

- [PDF Slides](deepfakes.pdf)
- [ODP Slides](deepfakes.odp)

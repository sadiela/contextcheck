# Frontend Documentation

## Page Breakdown

### Homepage

**Component Responsibilities:** 
- Renders the HeaderWrapper, InputWrapper, Results components
- Handles axios API calls to our backend, and communicates the responses with the 3 components mentioned above

#### Child Components:
- **HeaderWrapper:** renders the nav bar and the how-to-use steps
- **InputWrapper:** handles input steps and displays axios errors (if any)
  - **StepZero:** Prompts user to choose between plaintext / url input
  - **StepOne:** Where user enters plaintext / url depending on StepZero. Error checks the input. Displays possible urls.
  - **Loader:** The loading indicator
- **Results:** Renders BiasIndicator, TextPane (and MetaWrapper, Related Articles if url)
  - **BiasIndicator:** Displays bias in form of a sentence, allows user to open a modal to explain that sentence.
  - **TextPane:** Displays all text. Red words score above a certain threshold and display part of speech, score on hover
  - **MetaWrapper:** Constructs a table with Title, Author, Date, Source Tags
  - **RelatedArticles:** Constructs a table with related article links, title, source

---

### About Us

---

### Deets

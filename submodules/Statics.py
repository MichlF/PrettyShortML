from pathlib import Path
from time import time

import numpy as np
import requests
from langdetect import detect_langs


class Statics:
    """
    Collection of staticmethods that do not fit semantically to any of the other classes (yet).
    """

    @staticmethod
    def crawl_get_website_content(
        url: str,
        path_save: str = None,
        verbose: bool = False,
        t_wait: float = np.random.uniform(1, 3),
        *args,
        **kwargs,
    ) -> str:
        """
        Simple webcrawler method that requests and, if desired, stores the html
        content of a given website.

        Parameters
        ----------
        url : str
            URL to the website.
        path_save : str, optional
            Path to save the content of the website. If None provided,
            content is not saved, by default None.
        verbose : bool, optional
            Whether or not status code should always be printed or just
            for errors, by default False.
        t_wait : float, optional
            Time, in seconds, to wait before sending the request, by default.
            Random float between 1 and 3. Meant to reduce the likelihood
            of getting banned or timed out by requested server.
        *args / *kwargs
            Will be passed to the pathlib.write_text() function.

        Returns
        -------
        request.text : str
            Website content.
        """
        if t_wait:
            time.sleep(t_wait)
        try:
            req = requests.get(url)
            if verbose:
                print(f"Website status code: {req.status_code}")
            if (req.status_code >= 400) and (req.status_code < 500):
                print(
                    "\nERROR: User authorization or input error. Server"
                    f" response: {req.status_code}"
                )
            elif (req.status_code >= 500) and (req.status_code < 600):
                print(
                    "\nERROR: Server-sided error. Server response:"
                    f" {req.status_code}"
                )
            if path_save:
                try:
                    path_save = Path(
                        str(Path(".").resolve()) + "/" + path_save
                    )  # this is nasty, but VSC seems to mess up pathlib functions
                    path_save.parent.mkdir(exist_ok=True, parents=True)
                    path_save.write_text(req.text, *args, **kwargs)
                except Exception as _e:
                    print(_e, "\nERROR: Website could not be written to file.")

            return req.text
        except Exception as _e:
            print(_e, "\nERROR: Could not request url !")

    @staticmethod
    def nlp_get_lang_proba(text: str, language: str = "en") -> float|str:
        """
        Detects the likelihood a given string is in the specified language.

        Parameters
        ----------
        text : str
            The text string that should be analyzed.
        language : str, optional
            The language code for the analyzer, by default "en" for English.

        Returns
        -------
        float
            Returns the probability that the text is in the language specified.
        """
        detections = detect_langs(text)
        if not detections:
            return "NaN"
        for detection in detections:
            if detection.lang == language:

                return detection.prob

        return 0.0

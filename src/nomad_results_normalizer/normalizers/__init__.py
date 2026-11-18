from nomad.config.models.plugins import NormalizerEntryPoint


class ResultsNormalizerEntryPoint(NormalizerEntryPoint):
    """Entry point for normalizing simulation data into results.

    This normalizer populates results from supported simulation data in
    ``archive.data``.
    """

    level: int = 3

    def load(self):
        """Load the ResultsNormalizer, creating it dynamically to avoid
        circular imports.

        CIRCULAR IMPORT WORKAROUND:
        This method is called AFTER nomad.normalizing has finished initializing,
        so it's safe to import from nomad.normalizing here.

        We create the ResultsNormalizer class dynamically using type() to
        combine:
        1. Proper inheritance from nomad.normalizing.Normalizer (for NOMAD
           plugin system)
        2. Implementation from ResultsNormalizerBase (which doesn't inherit
           to avoid circular import)

        This pattern ensures:
        - No circular imports during module loading
        - Proper isinstance() checks in NOMAD's entry point validation
        - All methods from ResultsNormalizerBase are available
        """
        # Import base Normalizer first (safe - load() runs after nomad.normalizing init)
        from nomad.normalizing import Normalizer as BaseNormalizer

        # Import implementation (plain class without Normalizer inheritance)
        from nomad_results_normalizer.normalizers.results import ResultsNormalizerBase

        # Create the actual class with proper inheritance using type()
        # This combines BaseNormalizer (proper base class) with
        # ResultsNormalizerBase (implementation)
        ResultsNormalizer = type(
            'ResultsNormalizer',
            (BaseNormalizer,),  # Inherit from nomad.normalizing.Normalizer
            {
                k: v
                for k, v in ResultsNormalizerBase.__dict__.items()
                if not k.startswith(
                    '__'
                )  # Skip only dunder methods, keep private methods
            },  # Copy all methods, including private implementation helpers.
        )

        return ResultsNormalizer()


results_normalizer_plugin = ResultsNormalizerEntryPoint(
    name='Results',
    description='Results normalizer for simulation data in archive.data.',
)

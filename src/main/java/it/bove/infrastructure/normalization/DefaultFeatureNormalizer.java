package it.bove.infrastructure.normalization;

        import it.bove.domain.realestate.FeatureNormalizer;

        /**
         * Implementazione predefinita del normalizzatore di caratteristiche.
         */
        public class DefaultFeatureNormalizer implements FeatureNormalizer {
            private final double[] featureMin;
            private final double[] featureMax;

            /**
             * Costruttore con valori minimi e massimi per la normalizzazione.
             *
             * @param featureMin Array dei valori minimi per ciascuna caratteristica
             * @param featureMax Array dei valori massimi per ciascuna caratteristica
             */
            public DefaultFeatureNormalizer(double[] featureMin, double[] featureMax) {
                this.featureMin = featureMin;
                this.featureMax = featureMax;
            }

            @Override
            public double[] normalize(double[] features) {
                double[] normalized = new double[features.length];
                for (int i = 0; i < features.length; i++) {
                    normalized[i] = (features[i] - featureMin[i]) / (featureMax[i] - featureMin[i]);
                }
                return normalized;
            }

            @Override
            public double[] denormalize(double[] normalizedFeatures) {
                double[] denormalized = new double[normalizedFeatures.length];
                for (int i = 0; i < normalizedFeatures.length; i++) {
                    denormalized[i] = normalizedFeatures[i] * (featureMax[i] - featureMin[i]) + featureMin[i];
                }
                return denormalized;
            }
        }
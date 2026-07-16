package it.bove.infrastructure.normalization;

import it.bove.domain.realestate.PriceNormalizer;

/**
      * Implementazione predefinita del normalizzatore di prezzi.
      */
     public class DefaultPriceNormalizer implements PriceNormalizer {
         private final double priceMin;
         private final double priceMax;

         /**
          * Costruttore con prezzi minimi e massimi per la normalizzazione.
          *
          * @param priceMin Prezzo minimo in euro
          * @param priceMax Prezzo massimo in euro
          */
         public DefaultPriceNormalizer(double priceMin, double priceMax) {
             this.priceMin = priceMin;
             this.priceMax = priceMax;
         }

         @Override
         public Double normalize(Double price) {
             return (price - priceMin) / (priceMax - priceMin);
         }

         @Override
         public Double denormalize(Double normalizedPrice) {
             return normalizedPrice * (priceMax - priceMin) + priceMin;
         }
     }
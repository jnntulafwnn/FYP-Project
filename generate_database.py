import pandas as pd

# Create comprehensive Malaysian item database
items_data = {
    'Item Name': [
        # Personal Care
        'Sabun Dettol (900ml)', 'Sabun Lifebuoy (850ml)', 'Syampu Pantene (400ml)',
        'Syampu Head & Shoulders (600ml)', 'Ubat Gigi Colgate (175g)', 'Ubat Gigi Darlie (250g)',
        'Bedak Johnson (200g)', 'Minyak Wangi Gatsby', 'Losyen Vaseline (400ml)',
        'Tissue Kotak Kleenex', 'Tissue Gulung Scott (10 roll)', 'Pampers M (50pcs)',
        
        # Cooking Oil & Condiments
        'Minyak Masak Seri (5kg)', 'Minyak Masak Buruh (3kg)', 'Minyak Zaitun Bertolli (500ml)',
        'Kicap Masin Cap Kipas (600ml)', 'Kicap Manis Habhal (340ml)', 'Sos Tiram Lee Kum Kee (510g)',
        'Sos Cili Maggi (340g)', 'Sos Tomato Life (325g)', 'Cuka Halagel (450ml)',
        'Garam Kasar Cap Saga (1kg)', 'Gula Putih CSR (1kg)', 'Gula Perang Gula Prai (500g)',
        
        # Rice & Grains
        'Beras Faiza (10kg)', 'Beras Jati (5kg)', 'Beras Basmati India Gate (2kg)',
        'Mihun Kering AAA (400g)', 'Mee Kuning  Wantan (500g)', 'Bihun Gandum Adabi (200g)',
        
        # Canned & Packaged
        'Sardin Ayam Brand (230g)', 'Tuna Cili King\'s Cup (150g)', 'Susu Pekat Manis F&N (390g)',
        'Susu Segar Dutch Lady (1L)', 'Kopi Cap Kapal Api (200g)', 'Teh Boh Cameron (100 bags)',
        'Milo Australia (1kg)', 'Nescafe Original (200g)', 'Maggi Kari (5 packs)',
        
        # Snacks
        'Mamee Monster (25pcs)', 'Twisties Cheese (65g)', 'Roti Gardenia Jumbo',
        'Biskut Tiger (300g)', 'Kuaci Rota (150g)', 'Keropok Ikan AAA (200g)',
        
        # Fresh Produce (estimated)
        'Ayam Kampung (1kg)', 'Daging Lembu Australia (1kg)', 'Ikan Kembung (1kg)',
        'Udang Galah (500g)', 'Telur Gred A (10 biji)', 'Telur Ayam Kampung (10 biji)',
        'Bawang Merah India (1kg)', 'Bawang Putih China (500g)', 'Cili Merah (250g)',
        'Kentang Belanda (1kg)', 'Tomato (1kg)', 'Timun (3 biji)',
        
        # Vegetables
        'Bayam (1 ikat)', 'Kangkung (1 ikat)', 'Sawi (1 ikat)',
        'Kobis Bulat (1 biji)', 'Kacang Panjang (500g)', 'Carrot Cameron (500g)',
        'Terung Ungu (500g)', 'Bendi (300g)', 'Taugeh (500g)',
        
        # Fruits
        'Pisang Berangan (1 sisir)', 'Epal Fuji New Zealand (4 biji)', 'Oren Australia (1kg)',
        'Anggur Hitam (500g)', 'Tembikai (1 biji)', 'Mangga Harum Manis (3 biji)',
        'Durian Musang King (1kg)', 'Rambutan (1kg)', 'Betik Hong Kong (1 biji)',
        
        # Frozen Food
        'Nugget Ayam Ayamas (850g)', 'Sosej Ramly (480g)', 'Ikan Bilis Besar (500g)',
        'Udang Vannamei Frozen (800g)', 'Sos Pizza Prego (300g)',
        
        # Beverages
        'Air Mineral Spritzer (1.5L)', 'Air Kotak Ribena (1L)', '100 Plus Can (24 cans)',
        'Coca Cola (1.5L)', 'Pepsi (1.5L)', 'Heaven & Earth Green Tea (1.5L)',
        'Sirap Ros Sunquick (840ml)', 'Soya Bean F&N (1L)',
        
        # Household
        'Pencuci Pinggan Mama Lemon (900ml)', 'Sabun Cuci Baju Breeze (4kg)',
        'Pencuci Lantai Cif (2L)', 'Pewangi Baju Downy (900ml)', 'Penyegar Udara Glade',
        'Ubat Nyamuk Bakar Fumakilla (10 coil)', 'Racun Ridsect Spray', 'Sarung Tangan Getah Marigold',
        
        # Dairy
        'Keju Cheddar Kraft (180g)', 'Mentega Buttercup (250g)', 'Yogurt Dutch Lady (140g)',
        'Marjerin Planta (480g)',
        
        # Additional Cooking Items
        'Tepung Gandum Cap Sauh (1kg)', 'Tepung Jagung Clarico (400g)', 'Tepung Beras Rose (500g)',
        'Serbuk Kari Adabi (250g)', 'Jintan Manis (100g)', 'Bunga Lawang (50g)',
        'Asam Jawa (200g)', 'Belacan Cap Udang (200g)', 'Cendawan Shitake Kering (100g)',
        
        # Instant Food
        'Mi Sedaap Goreng (5 packs)', 'Cintan Kari (5 packs)', 'Bihun Sup Adabi (6 packs)',
        'Bubur Nasi Adabi (80g)', 'Sup Tulang Segera A1 (60g)',
        
        # Bakery
        'Roti Massimo Wholemeal', 'Cream Crackers Hup Seng (428g)', 'Biskut Marie Khong Guan',
        'Wafer Nissin (140g)', 'Kek Coklat Cadbury (300g)',
        
        # Herbs & Spices
        'Daun Kari Segar (1 ikat)', 'Halia Muda (200g)', 'Lengkuas (150g)',
        'Serai Wangi (1 ikat)', 'Daun Limau Purut (50g)', 'Kunyit Hidup (200g)',
        
        # Baby Products
        'Pampers Pants L (68 pcs)', 'Mamypoko S (90 pcs)', 'Baby Wipes Pigeon (80 sheets)',
        'Susu Dumex (900g)', 'Bubur Bayi Nestle (120g)', 'Botol Susu Avent',
        
        # Health & Wellness
        'Vitamin C Blackmores (60 tabs)', 'Minyak Gamat (60ml)', 'Panadol Actifast (20 tabs)',
        'Plaster Hansaplast', 'Ubat Batuk Cap Ibu Dan Anak (100ml)',
        
        # Seasonal/Festival Items  
        'Kurma Mesir (500g)', 'Lemang Instant (1 pack)', 'Rendang Paste Adabi (200g)',
        'Kuih Raya Mix', 'Mandarin Oren CNY',
        
        # Pet Food
        'Makanan Kucing Whiskas (1.2kg)', 'Makanan Anjing Pedigree (1.5kg)',
        
        # Miscellaneous Groceries
        'Santan Kotak Ayam Brand (200ml)', 'Agar-agar Swallow (10g)', 'Serbuk Ovalette (50g)',
        'Pewarna Makanan', 'Essence Vanila', 'Perasa Pandan',
        'Kacang Tanah Goreng (200g)', 'Bijan Putih (100g)', 'Kelapa Parut (200g)',
        
        # Additional Personal Care
        'Syampu Rejoice (340ml)', 'Pencuci Muka Garnier (100ml)', 'Losyen Muka Olay (50ml)',
        'Deodorant Rexona Roll-On', 'Kapas Muka Watsons (200 pcs)',
        
        # More Beverages
        'Teh Tarik Boh (10 sachet)', 'Kopi White Coffee Oldtown (15 sachet)',
        'Minuman Isotonic Revive (500ml)', 'Jus Kotak Marigold (1L)',
        
        # More Snacks
        'Kacang Putih Gardenia', 'Muruku Murruku (200g)', 'Keropok Lekor (300g)',
        'Dodol Durian (250g)', 'Halwa Maskat (200g)',
        
        # More Frozen
        'Karipap Pusing Frozen (20 pcs)', 'Popiah Goreng Frozen (10 pcs)',
        'Ikan Tenggiri Fillet (500g)', 'Sotong Tube (400g)',
        
        # Additional Household
        'Berus Tandas Mr Muscle', 'Pembersih Kaca Magiclean', 'Lap Microfiber',
        'Pewangi Kereta Little Trees', 'Penyedut Habuk Filter',
        
        # More Cooking Essentials
        'Sos Inglish Lea & Perrins', 'Sos Blackpepper Kimball', 'Rempah Kari Ikan Adabi',
        'Serbuk Kunyit (100g)', 'Lada Hitam Kisar (50g)', 'Jintan Putih (100g)',
        
        # Additional Fresh Produce
        'Labu Manis (1kg)', 'Kacang Botol (500g)', 'Daun Sawi (1 ikat)',
        'Cendawan Butang (200g)', 'Lobak Merah (500g)', 'Brokoli (1 biji)',
        
        # More Fruits
        'Nanas Madu (1 biji)', 'Jambu Batu Merah (500g)', 'Belimbing Manis (300g)',
        'Limau Nipis (10 biji)', 'Kelapa Muda (1 biji)',
        
        # Additional Dairy & Eggs
        'Keju Parmesan (100g)', 'Krim Masak Anchor (250ml)', 'Telur Puyuh (20 biji)',
        
        # More Instant/Ready Food
        'Nasi Goreng Paste Adabi', 'Tom Yam Cube Knorr', 'Perencah Char Kuey Teow',
        
        # Bread & Pastry
        'Roti Canai Frozen (5 pcs)', 'Puff Pastry Kawan', 'Roti Paratha Frozen',
        
        # More Condiments
        'Oyster Sauce Maekrua', 'Pes Tomyam Lobo', 'Sweet Chili Sauce Mae Ploy',
        'Hoisin Sauce Lee Kum Kee', 'Sesame Oil Kadoya (200ml)',
        
        # Additional Snacks & Treats
        'Coklat Cadbury Dairy Milk', 'Gula-gula Sugus', 'Permen Kopiko (100g)',
        'Jelly Cup Mogu Mogu', 'Ice Cream Wall\'s Cornetto',
        
        # More Health Items
        'Minyak Angin Cap Kapak', 'Minyak Gamat Gold-G', 'Ubat Sakit Perut Cap Lang',
        
        # Additional Baby Care
        'Bedak Bayi Johnson', 'Minyak Telon Konicare', 'Baby Shampoo Pigeon',
        
        # More Household Cleaning
        'Pembersih Toilet Harpic', 'Sabun Cuci Pinggan Sunlight', 'Pengharum Bilik Glade',
        'Plastik Sampah Hefty (50 pcs)',
        
        # Extra Cooking Ingredients
        'Sos Fish Megachef', 'Tauco Kwong Cheong Thye', 'Udang Kering (100g)',
        'Ikan Masin Bulu Ayam (200g)', 'Sup Bunjut Powder',
        
        # More Fresh Items
        'Peria Katak (300g)', 'Daun Kesum (1 ikat)', 'Bunga Kantan (5 biji)',
        'Petai (1 papan)', 'Jering (500g)',
        
        # Additional Fruits
        'Langsat (1kg)', 'Duku (500g)', 'Cempedak (1 biji)',
        'Markisa (5 biji)', 'Buah Naga Merah (2 biji)',
        
        # Specialty Items
        'Madu Kelulut (250ml)', 'Kurma Ajwa (500g)', 'Susu Kambing Segar (1L)',
        
        # More Beverages
        'Air Kelapa Prang (1L)', 'Barley Yeos (300ml)', 'Chrysanthemum Tea (1L)',
        
        # Additional Frozen Seafood
        'Ketam Bunga Frozen (500g)', 'Kerang (400g)', 'Lala (300g)',
        
        # More Personal Care
        'Bedak Sejuk Cap Limau', 'Minyak Rambut Brylcreem', 'Cologne Pucelle',
        
        # More Baking
        'Coklat Emulco (100ml)', 'Esen Strawberi', 'Baking Powder (50g)',
        'Soda Bikarbonat (100g)', 'Krim Tartar (50g)',
        
        # Last Items to reach 300
        'Perencah Ayam Gunting', 'Sos Satay Segera', 'Cincau Hitam (200g)',
        'Sambal Nyet Berapi', 'Pes Laksa Johor', 'Rempah Sup Tulang',
        'Kerisik (100g)', 'Pes Assam Pedas', 'Perencah Nasi Ayam',
        'Sos Black Pepper', 'Serbuk Lada Sulah', 'Jintan Kasar',
        'Halba (50g)', 'Ketumbar Biji (100g)', 'Kayu Manis (50g)',
        'Bunga Cengkih (30g)', 'Pelat Hitam (50g)', 'Sos Tiram Merah',
        'Pes Tomyam Thai', 'Pes Rendang Tok',
    ],
    
    'Category': [
        # Personal Care (12)
        'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care',
        'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care',
        
        # Cooking Oil & Condiments (12)
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        
        # Rice & Grains (6)
        'Rice & Grains', 'Rice & Grains', 'Rice & Grains', 'Rice & Grains', 'Rice & Grains', 'Rice & Grains',
        
        # Canned & Packaged (9)
        'Canned & Packaged', 'Canned & Packaged', 'Canned & Packaged', 'Canned & Packaged',
        'Beverages', 'Beverages', 'Beverages', 'Beverages', 'Instant Food',
        
        # Snacks (6)
        'Snacks', 'Snacks', 'Bakery', 'Snacks', 'Snacks', 'Snacks',
        
        # Fresh Produce (12)
        'Meat & Seafood', 'Meat & Seafood', 'Meat & Seafood', 'Meat & Seafood',
        'Fresh Produce', 'Fresh Produce', 'Fresh Produce', 'Fresh Produce', 'Fresh Produce',
        'Fresh Produce', 'Fresh Produce', 'Fresh Produce',
        
        # Vegetables (9)
        'Fresh Produce', 'Fresh Produce', 'Fresh Produce', 'Fresh Produce', 'Fresh Produce',
        'Fresh Produce', 'Fresh Produce', 'Fresh Produce', 'Fresh Produce',
        
        # Fruits (9)
        'Fruits', 'Fruits', 'Fruits', 'Fruits', 'Fruits', 'Fruits', 'Fruits', 'Fruits', 'Fruits',
        
        # Frozen Food (5)
        'Frozen Food', 'Frozen Food', 'Frozen Food', 'Frozen Food', 'Cooking Essentials',
        
        # Beverages (8)
        'Beverages', 'Beverages', 'Beverages', 'Beverages', 'Beverages', 'Beverages', 'Beverages', 'Beverages',
        
        # Household (8)
        'Household', 'Household', 'Household', 'Household', 'Household', 'Household', 'Household', 'Household',
        
        # Dairy (4)
        'Dairy', 'Dairy', 'Dairy', 'Dairy',
        
        # Additional Cooking Items (9)
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        
        # Instant Food (5)
        'Instant Food', 'Instant Food', 'Instant Food', 'Instant Food', 'Instant Food',
        
        # Bakery (5)
        'Bakery', 'Bakery', 'Bakery', 'Bakery', 'Bakery',
        
        # Herbs & Spices (6)
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        
        # Baby Products (6)
        'Baby Care', 'Baby Care', 'Baby Care', 'Baby Care', 'Baby Care', 'Baby Care',
        
        # Health & Wellness (5)
        'Health & Wellness', 'Health & Wellness', 'Health & Wellness', 'Health & Wellness', 'Health & Wellness',
        
        # Seasonal/Festival Items (5)
        'Special Occasions', 'Special Occasions', 'Cooking Essentials', 'Special Occasions', 'Special Occasions',
        
        # Pet Food (2)
        'Pet Care', 'Pet Care',
        
        # Miscellaneous Groceries (9)
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        'Cooking Essentials', 'Cooking Essentials', 'Snacks', 'Cooking Essentials', 'Cooking Essentials',
        
        # Additional Personal Care (5)
        'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care',
        
        # More Beverages (4)
        'Beverages', 'Beverages', 'Beverages', 'Beverages',
        
        # More Snacks (5)
        'Snacks', 'Snacks', 'Snacks', 'Snacks', 'Snacks',
        
        # More Frozen (4)
        'Frozen Food', 'Frozen Food', 'Meat & Seafood', 'Meat & Seafood',
        
        # Additional Household (5)
        'Household', 'Household', 'Household', 'Household', 'Household',
        
        # More Cooking Essentials (6)
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        
        # Additional Fresh Produce (6)
        'Fresh Produce', 'Fresh Produce', 'Fresh Produce', 'Fresh Produce', 'Fresh Produce', 'Fresh Produce',
        
        # More Fruits (5)
        'Fruits', 'Fruits', 'Fruits', 'Fruits', 'Fruits',
        
        # Additional Dairy & Eggs (3)
        'Dairy', 'Dairy', 'Fresh Produce',
        
        # More Instant/Ready Food (3)
        'Instant Food', 'Cooking Essentials', 'Cooking Essentials',
        
        # Bread & Pastry (3)
        'Frozen Food', 'Frozen Food', 'Frozen Food',
        
        # More Condiments (5)
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        
        # Additional Snacks & Treats (5)
        'Snacks', 'Snacks', 'Snacks', 'Snacks', 'Snacks',
        
        # More Health Items (3)
        'Health & Wellness', 'Health & Wellness', 'Health & Wellness',
        
        # Additional Baby Care (3)
        'Baby Care', 'Baby Care', 'Baby Care',
        
        # More Household Cleaning (4)
        'Household', 'Household', 'Household', 'Household',
        
        # Extra Cooking Ingredients (5)
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        
        # More Fresh Items (5)
        'Fresh Produce', 'Cooking Essentials', 'Cooking Essentials', 'Fresh Produce', 'Fresh Produce',
        
        # Additional Fruits (5)
        'Fruits', 'Fruits', 'Fruits', 'Fruits', 'Fruits',
        
        # Specialty Items (3)
        'Special Occasions', 'Special Occasions', 'Dairy',
        
        # More Beverages (3)
        'Beverages', 'Beverages', 'Beverages',
        
        # Additional Frozen Seafood (3)
        'Meat & Seafood', 'Meat & Seafood', 'Meat & Seafood',
        
        # More Personal Care (3)
        'Personal Care', 'Personal Care', 'Personal Care',
        
        # More Baking (5)
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        
        # Last Items to reach 300 (20)
        'Cooking Essentials', 'Cooking Essentials', 'Beverages', 'Cooking Essentials', 'Cooking Essentials',
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
        'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials',
    ]
}

# Create DataFrame
df = pd.DataFrame(items_data)

# Save to Excel
df.to_excel('item_database.xlsx', index=False)
print(f"✅ Item database created with {len(df)} items!")
print(f"Categories: {df['Category'].unique()}")

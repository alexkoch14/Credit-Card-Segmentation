# Credit Card Segmentation
Credit Card Segmentation Profitability Analysis.

# Problem Definition
The aftermath of the COVID-19 pandemic’s monetary policies, issued after decades of unsustainably low-interest rates to fuel liquidity markets, have begun to transpire.  Consumer inflation rates have skyrocketed to levels unseen since the 20th century and federal banks around the globe are doing everything in their power to tame this issue by increasing interest rates. 

While good for combatting inflation, high-interest rates create a recessionary environment by hampering economic activity. These effects are felt by borrowers, lenders, and the economy as a whole. They are superficially beneficial to banks since higher funds rates increase the yield on cash reserves from consumer and business activities, which go directly to net earnings. But these gains are nullified by an overall weak economy that elicits lower consumer loan demands and higher loss rates. 

According to the U.S. Federal Reserve, credit card debt consisted of 5.3% of all outstanding consumer debt in 2022, amounting to $841B dollars. Their higher interest rates relative to secured loans (mortgage, car loan, etc.) make them a big portion of a lender’s revenue despite lower exposure. As credit card loans remain to be lucrative, the objective is to provide card lenders with a risk-driven framework in order to maximize revenue and foremost minimize losses in the current economic landscape. Moreover, to modernize black box and outdated models currently used in industry and to bring transparency to applicant approval sizing.  

# Data Description 
### Datasets

The dataset is comprised of two files; an application record and a credit record. The former contains a list of every credit card applicant with demographic characteristics as features. Applicants are identified with a distinct client ID and 17 numeric & categorical attributes for income, gender, education type, etc. Likewise, the credit record uses client ID and Month on Book as primary keys and status as an attribute. For clarity, every row in the credit record represents a month that client X has been on file, and the status corresponds to the state of their loan at the end of that month. If the credit record contained 3 clients, each with 3 months of credit history, then the credit record would contain 9 rows (3 statuses per ID). Statuses can be “no loan”, “paid off”, “0-29 days overdue”, etc. up to “149+ days overdue”. Both files can be merged by client ID to join demographic information and credit history. 

### Target Attribute
Establishing a base of what drives credit card revenue is paramount in assessing performance since revenue optimization and loss provisioning are the desired scoring parameters. Credit cards make money through two ways: net interest income (~60%) and purchase volume (~40%). Net interest income represents the proceeds generated from overdue fees, and is derived using the following equation:

= Average Balance × Net Spread

= (#Active Accounts x Balance per Active)  × (Revolve Rate x Client Rate)

Average balance is defined as the product of total active accounts by the card balance per active, effectively representing the bank’s total exposure to the consumer market. Net spread is defined as the product of the percentage of active overdue clients (revolvers) by the nominal annual interest rate (19.99% APR industry standard). 

A typical statement cycle is 30 days, and the magnitude of net interest income is dependent on if/when the loan is paid off. In scenario 1, the client makes no transactions, and no interest revenue is generated. Similarly in scenario 2, the client pays their full balance by the due date and no interest income is generated. Scenario 3 is very lucrative as the client doesn’t pay their balance by the due date. Interest is charged on the outstanding balance and compounds for subsequent statement cycles until paid off. Scenario 4 is the adverse of scenario 3 wherein the client never pays off their loan, becomes a write off, and a substantial net loss to the issuer. As such, lenders must navigate the difficulty of expunging scenario 4s from their approvals while exploiting as many possible scenario 3s. 

The remaining 40% of revenue is generated from purchase volume and is derived using the following equation:

= Interchange + Fees

= (Purchase Volume x Interchange Rate) + (Additional Fees)

Interchange is defined as the product of all transactions made on the network each cycle ($ amount) by the interchange rate (2% industry standard). Interchange is a small fee paid by a merchant's bank (acquirer) to a cardholder's bank (issuer) to compensate the issuer for the benefits that merchants receive when they accept electronic payments. Additional fees correspond to ancillary charges such as annual fees, balance transfers, etc. 

# Findings
The model is capable of capturing approximately 3 of every 5 write-offs (~60% recall) while only misclassifying 1 in 5 good clients (~80% precision). This ratio is markedly superior to baseline results of 1 of every 5 write-offs (~20% recall). Hyperparameter tuning via Bayesian search yielded a 200% in model performance and findings underpin the difficultly in predicting client behaviour on a consistent risk adjusted basis. 

Client profitability is a function of spending behaviour, which is inherently entropic due to the unpredictably of human nature. Demography can be used to segment applicants, but individual performance can vary greatly between clients. Furthermore, historical credit data may not be in indicative of future spending habits since correlation doesn’t always imply causation. 


Nonetheless, the experiment provides insight into approval sizing decision boundaries by using a modernized white-box approach that surpasses the outdated legacy models used in industry. Demographic features of lower socio-economic class yield the highest predictive power. In fact, the top 7 reveal that single applicants with blue collar jobs are most likely to default on their credit loans. These clients are also less likely to have assets such as a phone, car, etc. 

The findings justify hunches commonly deduced by front line workers in the branch; less established single proletarians with no assets or dependents can afford jeopardizing their credit history for a quick loan. For clarity, it is discriminatory to deny an applicant based solely on instinct, but results indicate that human intuition isn’t always wrong. 

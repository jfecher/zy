#include "substitutingvisitor.h"
#include "antype.h"
#include "compiler.h"
#include "types.h"

using namespace std;

namespace ante {
    using namespace parser;

    /** Annotate all nodes with placeholder types */
    void SubstitutingVisitor::visit(RootNode *n){
        for(auto &m : n->imports)
            m->accept(*this);
        for(auto &m : n->types)
            m->accept(*this);
        for(auto &m : n->traits)
            m->accept(*this);
        for(auto &m : n->extensions)
            m->accept(*this);
        for(auto &m : n->funcs)
            m->accept(*this);

        for(auto &m : n->main){
            m->accept(*this);
        }
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(IntLitNode *n){}

    void SubstitutingVisitor::visit(FltLitNode *n){}

    void SubstitutingVisitor::visit(BoolLitNode *n){}

    void SubstitutingVisitor::visit(StrLitNode *n){}

    void SubstitutingVisitor::visit(CharLitNode *n){}

    void SubstitutingVisitor::visit(ArrayNode *n){
        for(auto &e : n->exprs)
            e->accept(*this);

        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(TupleNode *n){
        for(auto &e : n->exprs)
            e->accept(*this);

        if(!n->exprs.empty())
            n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(ModNode *n){
        if(n->expr)
            n->expr->accept(*this);
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(TypeNode *n){}

    void SubstitutingVisitor::visit(TypeCastNode *n){
        n->rval->accept(*this);
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(UnOpNode *n){
        n->rval->accept(*this);
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(SeqNode *n){
        for(auto &stmt : n->sequence){
            stmt->accept(*this);
        }
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(BinOpNode *n){
        n->lval->accept(*this);
        n->rval->accept(*this);
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(BlockNode *n){
        n->block->accept(*this);
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(RetNode *n){
        n->expr->accept(*this);
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(ImportNode *n){}

    void SubstitutingVisitor::visit(IfNode *n){
        n->condition->accept(*this);
        n->thenN->accept(*this);
        if(n->elseN){
            n->elseN->accept(*this);
            n->setType(applySubstitutions(substitutions, n->getType()));
        }
    }

    void SubstitutingVisitor::visit(NamedValNode *n){
        n->typeExpr->accept(*this);
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(VarNode *n){
        n->setType(applySubstitutions(substitutions, n->getType()));
    }


    void SubstitutingVisitor::visit(VarAssignNode *n){
        n->expr->accept(*this);
        n->ref_expr->accept(*this);

        n->ref_expr->setType(n->expr->getType());
        if(!n->modifiers.empty())
            n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(ExtNode *n){
        for(auto *m : *n->methods)
            m->accept(*this);
    }

    void SubstitutingVisitor::visit(JumpNode *n){
        n->expr->accept(*this);
    }

    void SubstitutingVisitor::visit(WhileNode *n){
        n->condition->accept(*this);
        n->child->accept(*this);
    }

    void SubstitutingVisitor::visit(ForNode *n){
        n->range->accept(*this);
        n->pattern->accept(*this);
        n->child->accept(*this);
    }

    void SubstitutingVisitor::visit(MatchNode *n){
        n->expr->accept(*this);
        for(auto &b : n->branches){
            b->accept(*this);
        }
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(MatchBranchNode *n){
        n->pattern->accept(*this);
        n->branch->accept(*this);
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(FuncDeclNode *n){
        for(auto *p : *n->params){
            p->accept(*this);
        }

        n->child->accept(*this);
        n->setType(applySubstitutions(substitutions, n->getType()));
    }

    void SubstitutingVisitor::visit(DataDeclNode *n){}

    void SubstitutingVisitor::visit(TraitNode *n){}
}